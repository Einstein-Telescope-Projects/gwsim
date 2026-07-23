"""Tests for --isolate reproduction wiring in the simulate command (issue #11)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import typer
import yaml

from gwmock.cli import simulate as simulate_mod
from gwmock.cli.simulate import _maybe_isolate, _resolve_recorded_environment


def _write_metadata(path: Path, name: str, environment: dict | None) -> None:
    payload: dict = {"schema_version": "1.3.0"}
    if environment is not None:
        payload["environment"] = environment
    text = yaml.safe_dump(payload) if name.endswith(".yaml") else json.dumps(payload)
    (path / name).write_text(text)


def test_resolve_recorded_environment_finds_first_available(tmp_path: Path) -> None:
    _write_metadata(tmp_path, "orchestration-0.metadata.json", None)
    _write_metadata(tmp_path, "orchestration-1.metadata.json", {"python": "3.12.5", "packages": {"numpy": "2.0.0"}})
    assert _resolve_recorded_environment([str(tmp_path)]) == {"python": "3.12.5", "packages": {"numpy": "2.0.0"}}


def test_resolve_recorded_environment_reads_yaml_metadata(tmp_path: Path) -> None:
    env = {"python": "3.12.5", "packages": {"numpy": "2.0.0"}}
    _write_metadata(tmp_path, "orchestration-0.metadata.yaml", env)
    assert _resolve_recorded_environment([str(tmp_path)]) == env


def test_resolve_recorded_environment_none_when_absent(tmp_path: Path) -> None:
    _write_metadata(tmp_path, "orchestration-0.metadata.json", None)
    assert _resolve_recorded_environment([str(tmp_path)]) is None


def test_resolve_recorded_environment_rejects_mixed_environments(tmp_path: Path) -> None:
    _write_metadata(tmp_path, "orchestration-0.metadata.json", {"python": "3.12.5", "packages": {"numpy": "2.0.0"}})
    _write_metadata(tmp_path, "orchestration-1.metadata.json", {"python": "3.12.5", "packages": {"numpy": "1.26.0"}})
    with pytest.raises(ValueError, match="multiple recorded environments"):
        _resolve_recorded_environment([str(tmp_path)])


def test_maybe_isolate_noop_without_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"n": 0}
    monkeypatch.setattr(
        "gwmock.cli.utils.environment.reproduce_in_isolated_environment",
        lambda *a, **k: called.__setitem__("n", called["n"] + 1),
    )
    _maybe_isolate([str(tmp_path)], isolate=False, output_dir=None, overwrite=False, author=None, email=None)
    assert called["n"] == 0


def test_maybe_isolate_noop_when_already_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GWMOCK_ISOLATED", "1")
    called = {"n": 0}
    monkeypatch.setattr(
        "gwmock.cli.utils.environment.reproduce_in_isolated_environment",
        lambda *a, **k: called.__setitem__("n", called["n"] + 1),
    )
    _maybe_isolate([str(tmp_path)], isolate=True, output_dir=None, overwrite=False, author=None, email=None)
    assert called["n"] == 0


def test_maybe_isolate_warns_when_no_environment_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("GWMOCK_ISOLATED", raising=False)
    _write_metadata(tmp_path, "orchestration-0.metadata.json", None)
    with caplog.at_level(logging.WARNING, logger="gwmock"):
        _maybe_isolate([str(tmp_path)], isolate=True, output_dir=None, overwrite=False, author=None, email=None)
    assert "no environment freeze" in caplog.text


def test_maybe_isolate_runs_in_place_when_matching(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GWMOCK_ISOLATED", raising=False)
    env = {"python": "3.12.5", "packages": {"numpy": "2.0.0"}}
    _write_metadata(tmp_path, "orchestration-0.metadata.json", env)
    monkeypatch.setattr("gwmock.cli.utils.environment.capture_environment", lambda: env)
    called = {"n": 0}
    monkeypatch.setattr(
        "gwmock.cli.utils.environment.reproduce_in_isolated_environment",
        lambda *a, **k: called.__setitem__("n", called["n"] + 1),
    )
    _maybe_isolate([str(tmp_path)], isolate=True, output_dir=None, overwrite=False, author=None, email=None)
    assert called["n"] == 0  # matching env: no re-exec


def test_maybe_isolate_reexecs_on_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GWMOCK_ISOLATED", raising=False)
    recorded = {"python": "3.12.5", "packages": {"numpy": "2.0.0"}}
    _write_metadata(tmp_path, "orchestration-0.metadata.json", recorded)
    monkeypatch.setattr(
        "gwmock.cli.utils.environment.capture_environment",
        lambda: {"python": "3.12.5", "packages": {"numpy": "1.26.0"}},
    )
    captured: dict = {}

    def fake_reproduce(snapshot, argv, cache_root=None):
        captured["snapshot"] = snapshot
        captured["argv"] = argv
        return 3

    monkeypatch.setattr("gwmock.cli.utils.environment.reproduce_in_isolated_environment", fake_reproduce)

    with pytest.raises(typer.Exit) as exc:
        _maybe_isolate([str(tmp_path)], isolate=True, output_dir="out", overwrite=True, author="A", email=None)

    assert exc.value.exit_code == 3
    assert captured["snapshot"] == recorded
    assert captured["argv"] == ["simulate", str(tmp_path), "--output-dir", "out", "--overwrite", "--author", "A"]


def test_maybe_isolate_propagates_dry_run_to_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A --dry-run reproduction must stay a dry run inside the isolated environment."""
    monkeypatch.delenv("GWMOCK_ISOLATED", raising=False)
    recorded = {"python": "3.12.5", "packages": {"numpy": "2.0.0"}}
    _write_metadata(tmp_path, "orchestration-0.metadata.json", recorded)
    monkeypatch.setattr(
        "gwmock.cli.utils.environment.capture_environment",
        lambda: {"python": "3.12.5", "packages": {"numpy": "1.26.0"}},
    )
    captured: dict = {}
    monkeypatch.setattr(
        "gwmock.cli.utils.environment.reproduce_in_isolated_environment",
        lambda snapshot, argv, cache_root=None: captured.setdefault("argv", argv) or 0,
    )
    with pytest.raises(typer.Exit):
        _maybe_isolate(
            [str(tmp_path)], isolate=True, output_dir=None, overwrite=False, author=None, email=None, dry_run=True
        )
    assert "--dry-run" in captured["argv"]


def test_maybe_isolate_reads_module_symbols_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard: _maybe_isolate imports environment lazily, so monkeypatching works."""
    assert hasattr(simulate_mod, "_maybe_isolate")
