"""Tests for environment capture and isolated-environment recreation (issue #11)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gwmock.cli.utils import environment as env_mod
from gwmock.cli.utils.environment import (
    ISOLATION_ENV_VAR,
    build_isolated_environment,
    capture_environment,
    diff_environment,
    environment_key,
    environment_requirements,
    environments_match,
    reproduce_in_isolated_environment,
)


def _snapshot(python: str = "3.12.5", **packages: str) -> dict:
    return {"python": python, "packages": dict(packages)}


# --- pure helpers ----------------------------------------------------------- #


def test_capture_environment_includes_python_and_self() -> None:
    env = capture_environment()
    assert env["python"].count(".") >= 2
    assert "gwmock" in env["packages"]


def test_environment_requirements_are_sorted_pins() -> None:
    reqs = environment_requirements(_snapshot(numpy="2.0.0", astropy="6.1.0"))
    assert reqs == ["astropy==6.1.0", "numpy==2.0.0"]


def test_diff_environment_reports_mismatch_and_missing_only() -> None:
    recorded = _snapshot(numpy="2.0.0", astropy="6.1.0")
    installed = _snapshot(numpy="1.26.0", scipy="1.14.0")  # numpy differs, astropy missing, scipy extra
    diff = diff_environment(recorded, installed)
    assert ("numpy", "2.0.0", "1.26.0") in diff
    assert ("astropy", "6.1.0", None) in diff
    assert all(name != "scipy" for name, _, _ in diff)  # extras ignored


def test_environments_match_requires_versions_and_python_minor() -> None:
    a = _snapshot("3.12.5", numpy="2.0.0")
    assert environments_match(a, _snapshot("3.12.9", numpy="2.0.0"))  # patch differs, minor same
    assert not environments_match(a, _snapshot("3.11.9", numpy="2.0.0"))  # minor differs
    assert not environments_match(a, _snapshot("3.12.5", numpy="1.26.0"))  # package differs
    # Extra installed packages break exactness (can change imports/plugins/entry points).
    assert not environments_match(a, _snapshot("3.12.5", numpy="2.0.0", scipy="1.14.0"))


def test_environment_key_is_stable_and_sensitive() -> None:
    base = _snapshot("3.12.5", numpy="2.0.0")
    assert environment_key(base) == environment_key(_snapshot("3.12.9", numpy="2.0.0"))  # patch ignored
    assert environment_key(base) != environment_key(_snapshot("3.12.5", numpy="2.0.1"))
    assert environment_key(base) != environment_key(_snapshot("3.11.5", numpy="2.0.0"))


# --- build_isolated_environment --------------------------------------------- #


def test_build_isolated_environment_creates_and_installs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, check=False, **_kwargs):
        calls.append(command)
        if "venv" in command:  # materialise the interpreter the builder expects
            env_dir = Path(command[command.index("venv") + 1])
            (env_dir / "bin").mkdir(parents=True, exist_ok=True)
            (env_dir / "bin" / "python").write_text("")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
    snapshot = _snapshot("3.12.5", numpy="2.0.0", astropy="6.1.0")

    python_bin = build_isolated_environment(snapshot, cache_root=tmp_path, uv_executable="/usr/bin/uv")

    assert python_bin.exists()
    venv_call = next(c for c in calls if "venv" in c)
    assert "--python" in venv_call
    assert "3.12" in venv_call
    install_call = next(c for c in calls if "install" in c)
    assert "numpy==2.0.0" in install_call
    assert "astropy==6.1.0" in install_call


def test_build_isolated_environment_reuses_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = {"n": 0}

    def fake_run(command, check=False, **_kwargs):
        runs["n"] += 1
        if "venv" in command:
            env_dir = Path(command[command.index("venv") + 1])
            (env_dir / "bin").mkdir(parents=True, exist_ok=True)
            (env_dir / "bin" / "python").write_text("")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
    snapshot = _snapshot("3.12.5", numpy="2.0.0")

    build_isolated_environment(snapshot, cache_root=tmp_path, uv_executable="/usr/bin/uv")
    runs_after_first = runs["n"]
    build_isolated_environment(snapshot, cache_root=tmp_path, uv_executable="/usr/bin/uv")

    assert runs["n"] == runs_after_first  # second call reused the cached env, ran nothing


def test_build_isolated_environment_without_uv_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env_mod.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="uv is required"):
        build_isolated_environment(_snapshot(), cache_root=tmp_path)


# --- reproduce_in_isolated_environment -------------------------------------- #


def test_reproduce_reexecs_child_with_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    python_bin = tmp_path / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("")
    monkeypatch.setattr(env_mod, "build_isolated_environment", lambda *a, **k: python_bin)

    captured: dict = {}

    def fake_run(command, env=None, check=False, **_kwargs):
        captured["command"] = command
        captured["env"] = env
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(env_mod.subprocess, "run", fake_run)

    code = reproduce_in_isolated_environment(_snapshot(), ["simulate", "metadata/", "--isolate", "--overwrite"])

    assert code == 7
    assert captured["command"] == [str(tmp_path / "bin" / "gwmock"), "simulate", "metadata/", "--overwrite"]
    assert captured["env"][ISOLATION_ENV_VAR] == "1"
