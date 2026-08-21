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


def test_environment_requirements_reject_option_shaped_names() -> None:
    """A tampered metadata pin that looks like an installer option is refused."""
    with pytest.raises(ValueError, match="invalid package name"):
        environment_requirements({"packages": {"--index-url": "http://evil.example"}})


def test_environment_requirements_reject_bad_version() -> None:
    with pytest.raises(ValueError, match="invalid version"):
        environment_requirements({"packages": {"numpy": "2.0.0; rm -rf /"}})


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
    # Pins follow a "--" so uv can never parse a pin as an option.
    assert "--" in install_call
    assert install_call.index("--") < install_call.index("numpy==2.0.0")


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


# --- the parts a reproduction depends on but no test reached ----------------- #


class TestThePythonMinorVersion:
    """Everything keys off ``major.minor``: the cache key, the venv it asks uv for, and the
    equality gate. A patch release is deliberately not part of it."""

    @pytest.mark.parametrize(
        ("version", "expected"),
        [("3.12.5", "3.12"), ("3.12", "3.12"), ("3", "3"), ("3.13.0rc1", "3.13"), (None, None), ("", None)],
    )
    def test_it_is_the_first_two_fields(self, version, expected) -> None:
        assert env_mod._python_minor(version) == expected


class TestTheCacheRoot:
    def test_the_environment_variable_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(env_mod.CACHE_ENV_VAR, str(tmp_path / "elsewhere"))
        assert env_mod.default_cache_root() == tmp_path / "elsewhere"

    def test_without_it_the_cache_lives_under_the_home_directory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(env_mod.CACHE_ENV_VAR, raising=False)
        assert env_mod.default_cache_root() == Path.home() / ".cache" / "gwmock" / "reproduction-envs"

    def test_an_empty_override_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An exported-but-empty variable is not a path; treating it as one would build the
        environment in the current directory."""
        monkeypatch.setenv(env_mod.CACHE_ENV_VAR, "")
        assert env_mod.default_cache_root() == Path.home() / ".cache" / "gwmock" / "reproduction-envs"

    def test_the_default_is_used_when_no_root_is_passed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``cache_root=None`` means "wherever the default says", which is what the CLI passes."""
        monkeypatch.setenv(env_mod.CACHE_ENV_VAR, str(tmp_path / "cache"))

        def fake_run(command, check=False, **_kwargs):
            if "venv" in command:
                env_dir = Path(command[command.index("venv") + 1])
                (env_dir / "bin").mkdir(parents=True, exist_ok=True)
                (env_dir / "bin" / "python").write_text("")
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)

        python_bin = build_isolated_environment(_snapshot("3.12.5"), uv_executable="/usr/bin/uv")

        assert (tmp_path / "cache") in python_bin.parents


class TestFindingUv:
    def test_an_explicit_executable_is_used_without_searching(self, tmp_path: Path, monkeypatch) -> None:
        """The argument is a fallback for uv *not* being on the path, so it cannot depend on
        finding it there as well."""
        monkeypatch.setattr(env_mod.shutil, "which", lambda _name: None)

        calls: list[list[str]] = []

        def fake_run(command, check=False, **_kwargs):
            calls.append(command)
            if "venv" in command:
                env_dir = Path(command[command.index("venv") + 1])
                (env_dir / "bin").mkdir(parents=True, exist_ok=True)
                (env_dir / "bin" / "python").write_text("")
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)

        build_isolated_environment(_snapshot(), cache_root=tmp_path, uv_executable="/opt/bin/uv")

        assert calls[0][0] == "/opt/bin/uv"

    def test_the_path_is_searched_for_uv_by_name(self, tmp_path: Path, monkeypatch) -> None:
        looked_for: list[str] = []

        def fake_which(name):
            looked_for.append(name)
            return "/found/uv"

        monkeypatch.setattr(env_mod.shutil, "which", fake_which)

        def fake_run(command, check=False, **_kwargs):
            if "venv" in command:
                env_dir = Path(command[command.index("venv") + 1])
                (env_dir / "bin").mkdir(parents=True, exist_ok=True)
                (env_dir / "bin" / "python").write_text("")
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)

        build_isolated_environment(_snapshot(), cache_root=tmp_path)

        assert looked_for == ["uv"]


class TestWhenTheCachedEnvironmentIsTrusted:
    @staticmethod
    def _fake_run(calls: list[list[str]]):
        def fake_run(command, check=False, **_kwargs):
            calls.append(command)
            if "venv" in command:
                env_dir = Path(command[command.index("venv") + 1])
                (env_dir / "bin").mkdir(parents=True, exist_ok=True)
                (env_dir / "bin" / "python").write_text("")
            return SimpleNamespace(returncode=0)

        return fake_run

    def test_a_marker_without_an_interpreter_is_rebuilt(self, tmp_path: Path, monkeypatch) -> None:
        """Both have to be there. A build interrupted after the marker was written, or an
        environment someone deleted the interpreter from, would otherwise be handed back as ready
        and the reproduction would fail on a missing executable."""
        snapshot = _snapshot("3.12.5", numpy="2.0.0")
        env_dir = tmp_path / environment_key(snapshot)
        env_dir.mkdir(parents=True)
        (env_dir / ".gwmock-ready").write_text("ready\n")

        calls: list[list[str]] = []
        monkeypatch.setattr(env_mod.subprocess, "run", self._fake_run(calls))

        python_bin = build_isolated_environment(snapshot, cache_root=tmp_path, uv_executable="/usr/bin/uv")

        assert calls, "the environment was reused despite having no interpreter"
        assert python_bin.exists()

    def test_an_interpreter_without_a_marker_is_rebuilt(self, tmp_path: Path, monkeypatch) -> None:
        """The marker is the only record that the *install* finished, so an interpreter alone means
        a half-built environment."""
        snapshot = _snapshot("3.12.5", numpy="2.0.0")
        env_dir = tmp_path / environment_key(snapshot)
        (env_dir / "bin").mkdir(parents=True)
        (env_dir / "bin" / "python").write_text("")

        calls: list[list[str]] = []
        monkeypatch.setattr(env_mod.subprocess, "run", self._fake_run(calls))

        build_isolated_environment(snapshot, cache_root=tmp_path, uv_executable="/usr/bin/uv")

        assert calls, "a half-built environment was reused"

    def test_a_partial_build_is_removed_before_the_rebuild(self, tmp_path: Path, monkeypatch) -> None:
        snapshot = _snapshot("3.12.5", numpy="2.0.0")
        env_dir = tmp_path / environment_key(snapshot)
        (env_dir / "leftover").mkdir(parents=True)

        monkeypatch.setattr(env_mod.subprocess, "run", self._fake_run([]))

        build_isolated_environment(snapshot, cache_root=tmp_path, uv_executable="/usr/bin/uv")

        assert not (env_dir / "leftover").exists()

    def test_the_marker_is_only_written_once_the_install_succeeds(self, tmp_path: Path, monkeypatch) -> None:
        """Otherwise the next run reuses an environment whose packages were never installed."""
        snapshot = _snapshot("3.12.5", numpy="2.0.0")

        def fake_run(command, check=False, **_kwargs):
            if "venv" in command:
                env_dir = Path(command[command.index("venv") + 1])
                (env_dir / "bin").mkdir(parents=True, exist_ok=True)
                (env_dir / "bin" / "python").write_text("")
                return SimpleNamespace(returncode=0)
            raise env_mod.subprocess.CalledProcessError(1, command)

        monkeypatch.setattr(env_mod.subprocess, "run", fake_run)

        with pytest.raises(env_mod.subprocess.CalledProcessError):
            build_isolated_environment(snapshot, cache_root=tmp_path, uv_executable="/usr/bin/uv")

        assert not (tmp_path / environment_key(snapshot) / ".gwmock-ready").exists()

    def test_concurrent_builds_of_one_environment_are_serialised_on_its_own_lock(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Keyed by the environment, not by the cache root: two different reproductions must not
        wait for each other, and two of the same must not install over each other."""
        snapshot = _snapshot("3.12.5", numpy="2.0.0")
        locked: list[str] = []

        class _Lock:
            def __init__(self, path):
                locked.append(str(path))

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        monkeypatch.setattr(env_mod, "FileLock", _Lock)
        monkeypatch.setattr(env_mod.subprocess, "run", self._fake_run([]))

        build_isolated_environment(snapshot, cache_root=tmp_path, uv_executable="/usr/bin/uv")

        assert locked == [str(tmp_path / f"{environment_key(snapshot)}.lock")]


class TestTheEnvironmentKey:
    def test_it_is_sixteen_characters(self) -> None:
        """Short enough for a directory name, long enough that two environments do not collide."""
        assert len(environment_key(_snapshot("3.12.5", numpy="2.0.0"))) == 16

    def test_it_is_hexadecimal(self) -> None:
        key = environment_key(_snapshot("3.12.5", numpy="2.0.0"))
        assert set(key) <= set("0123456789abcdef")

    def test_a_patch_release_does_not_change_it(self) -> None:
        """The recreated venv is pinned to major.minor, so keying on the patch would fragment the
        cache for environments uv would build identically."""
        assert environment_key(_snapshot("3.12.5", numpy="2.0.0")) == environment_key(
            _snapshot("3.12.9", numpy="2.0.0")
        )

    def test_a_different_minor_changes_it(self) -> None:
        assert environment_key(_snapshot("3.12.5", numpy="2.0.0")) != environment_key(
            _snapshot("3.13.0", numpy="2.0.0")
        )

    def test_a_different_version_changes_it(self) -> None:
        assert environment_key(_snapshot("3.12.5", numpy="2.0.0")) != environment_key(
            _snapshot("3.12.5", numpy="2.0.1")
        )

    def test_an_extra_package_changes_it(self) -> None:
        assert environment_key(_snapshot("3.12.5", numpy="2.0.0")) != environment_key(
            _snapshot("3.12.5", numpy="2.0.0", scipy="1.14.0")
        )

    def test_the_order_the_packages_were_recorded_in_does_not(self) -> None:
        first = {"python": "3.12.5", "packages": {"numpy": "2.0.0", "astropy": "6.1.0"}}
        second = {"python": "3.12.5", "packages": {"astropy": "6.1.0", "numpy": "2.0.0"}}
        assert environment_key(first) == environment_key(second)

    def test_an_invalid_pin_is_refused_rather_than_hashed(self) -> None:
        with pytest.raises(ValueError, match="invalid package name"):
            environment_key({"python": "3.12.5", "packages": {"--index-url": "1.0"}})


class TestComparingEnvironments:
    def test_an_extra_installed_package_is_not_an_exact_match(self) -> None:
        """Extras change optional imports, backend resolution and entry points, so the exact gate
        refuses them even though the drift warning ignores them."""
        recorded = _snapshot("3.12.5", numpy="2.0.0")
        installed = _snapshot("3.12.5", numpy="2.0.0", extra="1.0")
        assert environments_match(recorded, installed) is False
        assert diff_environment(recorded, installed) == []

    def test_a_patch_difference_in_python_still_matches(self) -> None:
        assert environments_match(_snapshot("3.12.5", numpy="2.0.0"), _snapshot("3.12.9", numpy="2.0.0")) is True

    def test_a_missing_package_is_reported_with_no_installed_version(self) -> None:
        assert diff_environment(_snapshot("3.12.5", numpy="2.0.0"), _snapshot("3.12.5")) == [("numpy", "2.0.0", None)]

    def test_mismatches_are_reported_in_name_order(self) -> None:
        recorded = _snapshot("3.12.5", numpy="2.0.0", astropy="6.1.0", scipy="1.14.0")
        installed = _snapshot("3.12.5", numpy="1.0.0", astropy="5.0.0", scipy="1.0.0")
        assert [name for name, _, _ in diff_environment(recorded, installed)] == ["astropy", "numpy", "scipy"]

    def test_an_empty_snapshot_compares_as_empty_rather_than_raising(self) -> None:
        assert diff_environment({}, {}) == []
        assert environments_match({}, {}) is True


class TestCapturingTheEnvironment:
    def test_the_scan_is_reused_rather_than_repeated(self) -> None:
        """Cached on purpose: per-batch metadata writes would otherwise rescan every distribution."""
        assert capture_environment() is capture_environment()

    def test_package_names_are_recorded_in_lower_case_and_sorted(self) -> None:
        packages = capture_environment()["packages"]
        assert list(packages) == sorted(packages)
        assert all(name == name.lower() for name in packages)

    def test_the_interpreter_is_recorded(self) -> None:
        captured = capture_environment()
        assert captured["python"]
        assert captured["python_implementation"]
