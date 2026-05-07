"""Tests for simulator state helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

from gwmock.simulator.state import PopulationIterationState


def test_population_iteration_state_loads_legacy_checkpoint(tmp_path: Path):
    """Legacy population checkpoint files should restore the iteration cursor."""
    checkpoint = tmp_path / "population.yaml"
    checkpoint.write_text(
        yaml.safe_dump(
            {
                "population": {
                    "current_index": 3,
                    "injected_indices": [1, 2],
                    "segment_map": {0: [1], 2: [3, 4]},
                }
            }
        ),
        encoding="utf-8",
    )

    state = PopulationIterationState(str(checkpoint))

    assert state.checkpoint_file == checkpoint
    assert state.current_index == 3
    assert state.injected_indices == [1, 2]
    assert state.segment_map == {0: [1], 2: [3, 4]}


def test_population_iteration_state_serializes_legacy_checkpoint_shape():
    """Serialization should preserve the pre-existing population checkpoint schema."""
    state = PopulationIterationState()
    state.current_index = 4
    state.injected_indices = [0, 2]
    state.segment_map = {1: [3, 5]}

    assert state.to_checkpoint_data() == {
        "population": {
            "current_index": 4,
            "injected_indices": [0, 2],
            "segment_map": {1: [3, 5]},
        }
    }


def test_population_iteration_state_ignores_invalid_checkpoint(tmp_path: Path):
    """Unreadable population checkpoints should fall back to a fresh state."""
    checkpoint = tmp_path / "population.yaml"
    checkpoint.write_text("population: [", encoding="utf-8")

    state = PopulationIterationState(checkpoint)

    assert state.current_index == 0
    assert state.injected_indices == []
    assert state.segment_map == {}
