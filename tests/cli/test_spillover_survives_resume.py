#
# Copyright (C) 2026 Leuven Gravity Institute
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
"""A resumed run must place the tail of a signal that crosses the resume point.

Measured on ``main`` with a 1.6+1.4 binary from 30 Hz across 32 s segments: interrupt after the
first batch checkpoints, re-run the same config, and the third frame -- the one holding the
**merger** -- comes back at a peak of exactly ``0.0`` against ``7.280e-23`` uninterrupted. The two
frames before it are bit-identical, so nothing about the run looks wrong; the loudest part of the
signal is simply absent.

The cause is that spillover lived only in memory. ``cached_data_chunks`` holds the part of a chunk
extending past the segment being built, and it was neither a ``StateAttribute`` nor saved anywhere
else, so a resumed process started with none of it.

Two things had to be right, and they fail independently:

* the checkpoint has to carry the chunks at all, and carry their ``metadata`` with them -- restoring
  the samples while dropping ``injection_parameters`` and ``event_id`` would look like a fix while
  losing the provenance the frames are indexed by;
* ``restore_batch_state`` has to put them back on the simulator.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.units.quantity import Quantity

from gwmock.cli.utils.checkpoint import CheckpointManager
from gwmock.data.time_series.time_series import TimeSeries
from gwmock.data.time_series.time_series_list import TimeSeriesList


def _chunk(start: float = 100.0, samples: int = 8, event_id: int = 3) -> TimeSeries:
    chunk = TimeSeries(
        data=np.arange(1.0, samples + 1.0).reshape(1, samples),
        start_time=Quantity(start, unit="s"),
        sampling_frequency=Quantity(8.0, unit="Hz"),
    )
    chunk.metadata.update({"injection_parameters": {"coa_time": start + 1.0}, "event_id": event_id})
    chunk[0].name = "H1:STRAIN"
    return chunk


class TestTheCheckpointCarriesSpillover:
    """The serialization half."""

    def test_samples_and_provenance_both_survive(self, tmp_path):
        """Both, because restoring one without the other is the failure that looks like success.

        ``to_json_dict`` carried neither ``metadata`` nor the channel identity, so a fix that only
        made the chunks reachable would have resumed with data that no longer says which signal it
        is -- and ``inject`` copies the channel name onto a tail deliberately, for the same reason.
        """
        manager = CheckpointManager(tmp_path)
        manager.save_checkpoint([0], "orchestration", 0, {"counter": 1}, TimeSeriesList([_chunk()]))

        restored = manager.get_last_simulator_spillover()

        assert len(restored) == 1
        np.testing.assert_array_equal(np.asarray(restored[0]).ravel(), np.arange(1.0, 9.0))
        assert restored[0].metadata["event_id"] == 3
        assert restored[0].metadata["injection_parameters"] == {"coa_time": 101.0}
        # `restored[0]` is the first *series*; its channels are indexed one level in.
        assert restored[0][0].name == "H1:STRAIN"
        assert float(restored[0].start_time.value) == 100.0

    def test_a_checkpoint_written_before_this_field_reads_as_no_spillover(self, tmp_path):
        """An upgrade mid-run must not turn a resumable checkpoint into a crash.

        ``None`` covers both "written by an older gwmock" and "that segment had no spillover", which
        are the same thing to the caller: there is nothing to carry in.
        """
        import json

        (tmp_path / "simulation.checkpoint.json").write_text(
            json.dumps(
                {
                    "completed_batch_indices": [0],
                    "last_simulator_name": "orchestration",
                    "last_completed_batch_index": 0,
                    "last_simulator_state": {"counter": 1},
                }
            )
        )

        assert CheckpointManager(tmp_path).get_last_simulator_spillover() is None


class TestRestorePutsSpilloverBack:
    """The wiring half, which the serialization test above cannot reach."""

    @staticmethod
    def _simulator_and_batch():
        from gwmock.cli.utils.simulation_plan import SimulationBatch

        class _Simulator:
            def __init__(self):
                self.cached_data_chunks = TimeSeriesList()
                self.counter = 1

            @property
            def state(self):
                return {"counter": self.counter}

            @state.setter
            def state(self, value):
                self.counter = value.get("counter", self.counter)

        batch = SimulationBatch.__new__(SimulationBatch)
        object.__setattr__(batch, "batch_index", 1)
        object.__setattr__(batch, "simulator_name", "orchestration")
        object.__setattr__(batch, "metadata", None)
        return _Simulator(), batch

    def test_the_simulator_gets_the_chunks(self):
        """Without this the checkpoint holds the tail and nothing ever reads it back."""
        from gwmock.cli.simulate_utils import restore_batch_state

        simulator, batch = self._simulator_and_batch()
        spillover = TimeSeriesList([_chunk()])

        restore_batch_state(simulator, batch, {"counter": 1}, spillover)

        assert len(simulator.cached_data_chunks) == 1
        assert simulator.cached_data_chunks[0].metadata["event_id"] == 3

    def test_no_spillover_leaves_the_simulator_alone(self):
        """A resumed run whose previous segment spilled nothing must not be handed an empty list.

        The distinction matters because ``None`` and ``TimeSeriesList()`` are both falsy: assigning
        unconditionally would work here and silently clobber chunks in any caller that had already
        populated them.
        """
        from gwmock.cli.simulate_utils import restore_batch_state

        simulator, batch = self._simulator_and_batch()
        simulator.cached_data_chunks = TimeSeriesList([_chunk(event_id=9)])

        restore_batch_state(simulator, batch, {"counter": 1}, None)

        assert len(simulator.cached_data_chunks) == 1
        assert simulator.cached_data_chunks[0].metadata["event_id"] == 9


@pytest.mark.integration
def test_a_resumed_run_keeps_the_merger():
    """The end-to-end statement, recorded here because the unit tests cannot make it.

    Not automated: it needs a real interrupt at a real checkpoint boundary, and the timing is
    process-dependent. Reproduced by hand on 2026-08-06 with the config in the investigation record,
    interrupting once ``.gwmock_checkpoints/simulation.checkpoint.json`` appears and re-running:

    ======================  ==============  =================  ===================
    frame                   uninterrupted   resumed on main    resumed with fix
    ======================  ==============  =================  ===================
    ...540                  1.286e-23       1.286e-23          1.286e-23
    ...572                  1.837e-23       1.837e-23          1.837e-23
    ...604 (merger)         7.280e-23       **0.000e+00**      7.280e-23
    ======================  ==============  =================  ===================

    Skipped rather than deleted so the procedure stays with the code it describes.
    """
    pytest.skip("manual: requires interrupting a real run at a checkpoint boundary")
