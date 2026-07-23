"""Unit tests for signal->frame lookup (issue #12, Part B)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from gwmock.cli.main import app
from gwmock.cli.simulate_utils import update_signal_index
from gwmock.cli.utils.signal_lookup import find_signals, parse_param_filter

runner = CliRunner()


def _write_batch(directory: Path, index: int, injections: list[dict], frames: list[str]) -> None:
    """Write a minimal batch metadata file plus its signal-index entry."""
    metadata = {
        "signal": {"injections": injections},
        "outputs": [{"kind": "signal", "path": frame} for frame in frames]
        + [{"kind": "noise", "path": f"noise-{index}.npy"}],
    }
    name = f"orchestration-{index}.metadata.json"
    (directory / name).write_text(json.dumps(metadata))
    update_signal_index(directory, metadata, name)


@pytest.fixture
def metadata_dir(tmp_path: Path) -> Path:
    _write_batch(
        tmp_path,
        0,
        [{"event_id": 0, "parameters": {"detector_frame_mass_1": 30.0, "coa_time": 100.5}}],
        ["signal/signal-0.gwf"],
    )
    _write_batch(
        tmp_path,
        1,
        [{"event_id": 1, "parameters": {"detector_frame_mass_1": 31.0, "coa_time": 104.5}}],
        ["signal/signal-1.gwf"],
    )
    return tmp_path


# --- parse_param_filter ----------------------------------------------------- #


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("mass_1>=30", ("mass_1", ">=", 30.0)),
        ("mass_1 <= 40", ("mass_1", "<=", 40.0)),
        ("coa_time>100.5", ("coa_time", ">", 100.5)),
        ("approximant==IMRPhenomXPHM", ("approximant", "==", "IMRPhenomXPHM")),
        ("n!=2", ("n", "!=", 2.0)),
    ],
)
def test_parse_param_filter(spec: str, expected: tuple) -> None:
    assert parse_param_filter(spec) == expected


def test_parse_param_filter_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="Invalid parameter filter"):
        parse_param_filter("mass_1")


# --- find_signals ----------------------------------------------------------- #


def test_signal_index_written(metadata_dir: Path) -> None:
    index = yaml.safe_load((metadata_dir / "signal_index.yaml").read_text())
    assert set(index) == {"0", "1"}
    assert index["0"]["frames"] == ["signal/signal-0.gwf"]
    assert index["0"]["coa_time"] == 100.5
    assert index["1"]["metadata"] == "orchestration-1.metadata.json"


def test_find_by_id_uses_index(metadata_dir: Path) -> None:
    matches = find_signals(metadata_dir, event_id=1)
    assert len(matches) == 1
    assert matches[0]["frames"] == ["signal/signal-1.gwf"]
    assert matches[0]["metadata"] == "orchestration-1.metadata.json"


def test_find_by_id_missing_returns_empty(metadata_dir: Path) -> None:
    assert find_signals(metadata_dir, event_id=99) == []


def test_find_by_param_filter_scans_metadata(metadata_dir: Path) -> None:
    matches = find_signals(metadata_dir, param_filters=[parse_param_filter("detector_frame_mass_1>=31")])
    assert [m["event_id"] for m in matches] == [1]
    assert matches[0]["frames"] == ["signal/signal-1.gwf"]
    assert matches[0]["parameters"]["detector_frame_mass_1"] == 31.0


def test_find_by_multiple_filters_are_anded(metadata_dir: Path) -> None:
    matches = find_signals(
        metadata_dir,
        param_filters=[parse_param_filter("detector_frame_mass_1>=30"), parse_param_filter("coa_time<104")],
    )
    assert [m["event_id"] for m in matches] == [0]


def test_signal_index_drops_stale_entries_on_rerun(tmp_path: Path) -> None:
    """Re-running a batch with different/no injections must not leave stale id rows."""
    # First run: batch 0 injects events 0 and 1.
    meta_a = {
        "signal": {
            "injections": [
                {"event_id": 0, "parameters": {"coa_time": 100.0}},
                {"event_id": 1, "parameters": {"coa_time": 101.0}},
            ]
        },
        "outputs": [{"kind": "signal", "path": "signal/signal-0.gwf"}],
    }
    update_signal_index(tmp_path, meta_a, "orchestration-0.metadata.json")
    assert set(yaml.safe_load((tmp_path / "signal_index.yaml").read_text())) == {"0", "1"}

    # Re-run the same batch, now injecting only event 0. Event 1's stale row must go.
    meta_b = {
        "signal": {"injections": [{"event_id": 0, "parameters": {"coa_time": 100.0}}]},
        "outputs": [{"kind": "signal", "path": "signal/signal-0.gwf"}],
    }
    update_signal_index(tmp_path, meta_b, "orchestration-0.metadata.json")
    index = yaml.safe_load((tmp_path / "signal_index.yaml").read_text())
    assert set(index) == {"0"}
    assert find_signals(tmp_path, event_id=1) == []


def test_signal_index_clears_entries_when_batch_reruns_with_no_injections(tmp_path: Path) -> None:
    """A batch that previously injected but now injects nothing clears its rows."""
    meta = {
        "signal": {"injections": [{"event_id": 5, "parameters": {"coa_time": 1.0}}]},
        "outputs": [{"kind": "signal", "path": "signal/signal-0.gwf"}],
    }
    update_signal_index(tmp_path, meta, "orchestration-0.metadata.json")
    assert "5" in yaml.safe_load((tmp_path / "signal_index.yaml").read_text())

    empty = {"signal": {"injections": []}, "outputs": []}
    update_signal_index(tmp_path, empty, "orchestration-0.metadata.json")
    assert yaml.safe_load((tmp_path / "signal_index.yaml").read_text()) == {}


def test_equality_filter_uses_tolerance(tmp_path: Path) -> None:
    """Numeric == tolerates representation noise; != is its complement."""
    _write_batch(
        tmp_path,
        0,
        [{"event_id": 0, "parameters": {"mass_1": 30.0}}],
        ["signal/signal-0.gwf"],
    )
    # Stored 30.0 vs a query value differing only by float noise still matches ==.
    assert [m["event_id"] for m in find_signals(tmp_path, param_filters=[("mass_1", "==", 30.0 + 1e-12)])] == [0]
    assert find_signals(tmp_path, param_filters=[("mass_1", "!=", 30.0 + 1e-12)]) == []


def test_find_id_and_param_combined(metadata_dir: Path) -> None:
    # event 0's mass is 30, so an id=0 + mass>=31 filter matches nothing.
    assert find_signals(metadata_dir, event_id=0, param_filters=[parse_param_filter("detector_frame_mass_1>=31")]) == []


# --- CLI -------------------------------------------------------------------- #


def test_cli_find_signal_by_id(metadata_dir: Path) -> None:
    result = runner.invoke(app, ["find-signal", "--metadata-dir", str(metadata_dir), "--id", "0"])
    assert result.exit_code == 0
    assert "signal/signal-0.gwf" in result.stdout
    assert "event 0" in result.stdout


def test_cli_find_signal_by_param_json(metadata_dir: Path) -> None:
    result = runner.invoke(
        app,
        ["find-signal", "--metadata-dir", str(metadata_dir), "--param", "detector_frame_mass_1>=31", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert [m["event_id"] for m in payload] == [1]


def test_cli_find_signal_requires_a_query(metadata_dir: Path) -> None:
    result = runner.invoke(app, ["find-signal", "--metadata-dir", str(metadata_dir)])
    assert result.exit_code != 0


def test_cli_find_signal_no_match_exits_nonzero(metadata_dir: Path) -> None:
    result = runner.invoke(app, ["find-signal", "--metadata-dir", str(metadata_dir), "--id", "99"])
    assert result.exit_code == 1
    assert "No matching signals" in result.stdout
