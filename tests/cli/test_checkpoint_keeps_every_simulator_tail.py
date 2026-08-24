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
"""A checkpoint must keep the tail of every simulator, not only the one that finished last.

A plan can execute several simulators. The checkpoint used to hold a single
``last_simulator_state``/``last_simulator_spillover`` pair, so once simulator B completed a batch,
simulator A's state and the tail of any signal crossing A's final segment boundary were overwritten
and gone from the file. Scoping the hand-over -- refusing to give A's tail to B -- prevents the
*dangerous* failure, real strain of the right shape landing in the wrong simulator's data, but it
cannot recover something that was never written down.

Both halves are the same bug and both are exercised here:

* **spillover**, which is lost silently -- a hole in the data where a signal crossing the boundary
  should have been continued;
* **state**, which is worse than lost, because the surviving one is *accepted*: the restore gate is
  ``batch_index == state["counter"]``, and a plan numbering its batches across simulators reaches
  that equality with the wrong simulator's state. A resumed run then puts A's RNG stream behind B's
  segment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gwmock.cli.simulate_utils import execute_plan
from gwmock.cli.utils.checkpoint import CheckpointManager
from gwmock.cli.utils.config import GlobalsConfig, SimulatorConfig, SimulatorOutputConfig
from gwmock.cli.utils.simulation_plan import SimulationBatch, SimulationPlan
from gwmock.simulator.base import Simulator
from gwmock.simulator.state import StateAttribute

#: Which batch of which simulator raises, keyed by simulator tag and read at ``simulate`` time.
#: The simulators are built inside ``execute_plan`` from their config, so a test cannot reach the
#: instance to arm it; and the second run of a resume needs the same class to succeed.
FAIL_AT_COUNTER: dict[str, int] = {}


@pytest.fixture(autouse=True)
def _disarm_failures():
    """No test leaves a failure armed for the next one."""
    FAIL_AT_COUNTER.clear()
    yield
    FAIL_AT_COUNTER.clear()


class TailSimulator(Simulator):
    """A simulator that carries a spillover tail and a state marker saying whose it is.

    ``tag`` is a state attribute on purpose: it travels with ``state``, so a segment generated
    after the wrong simulator's state was restored says so in its own metadata.
    """

    tag = StateAttribute(default="")

    def __init__(self, tag: str = "", **kwargs: Any):
        """Build the simulator.

        Args:
            tag: Name this simulator writes into its state, its data and its spillover.
            **kwargs: Passed to the base class.
        """
        super().__init__(**kwargs)
        self.tag = tag
        # Not state: spillover is raw samples, and `state` is copied into every metadata record.
        self.cached_data_chunks: list[str] = []

    def simulate(self) -> str:
        """Generate one batch, leaving a tail behind for the next one.

        Returns:
            A string identifying the simulator and the batch.

        Raises:
            RuntimeError: When this batch is the one the test arms to fail.
        """
        if FAIL_AT_COUNTER.get(str(self.tag)) == self.counter:
            raise RuntimeError(f"interrupted {self.tag} at batch {self.counter}")
        self.cached_data_chunks = [f"{self.tag}-tail-{self.counter}"]
        return f"{self.tag}-{self.counter}"

    def _save_data(self, data: Any, file_name: str | Path, **kwargs: Any) -> None:
        """Write the batch to a JSON file.

        Args:
            data: What ``simulate`` returned.
            file_name: Where to write it.
            **kwargs: Unused.
        """
        file_name = Path(file_name)
        file_name.parent.mkdir(parents=True, exist_ok=True)
        with file_name.open("w") as f:
            json.dump({"data": data, "tag": str(self.tag), "counter": self.counter}, f)


def _plan(checkpoint_directory: Path, segments: dict[str, list[int]]) -> SimulationPlan:
    """Build a plan running each named simulator over the batch indices given.

    Args:
        checkpoint_directory: Where the checkpoint goes -- inside the test's tmp_path, never the
            default relative ``checkpoints/``.
        segments: Simulator name to its batch indices. Indices are unique across simulators, as
            ``merge_plans`` numbers them, because ``completed_batch_indices`` is a flat set.

    Returns:
        The plan.
    """
    plan = SimulationPlan(checkpoint_directory=checkpoint_directory)
    for name, indices in segments.items():
        for index in indices:
            plan.add_batch(
                SimulationBatch(
                    simulator_name=name,
                    simulator_config=SimulatorConfig(
                        class_="tests.cli.test_checkpoint_keeps_every_simulator_tail.TailSimulator",
                        arguments={"tag": name},
                        output=SimulatorOutputConfig(file_name=f"{name}-{index}.json"),
                    ),
                    globals_config=GlobalsConfig(),
                    batch_index=index,
                )
            )
    return plan


def _run(plan: SimulationPlan, tmp_path: Path, expect_failure: bool) -> None:
    """Execute *plan* against the directories under *tmp_path*.

    Args:
        plan: What to run.
        tmp_path: The test's temporary directory.
        expect_failure: Whether the armed failure is expected to reach the caller.
    """
    directories = (plan, tmp_path / "output", tmp_path / "metadata")
    if expect_failure:
        with pytest.raises(RuntimeError, match="interrupted"):
            execute_plan(*directories, overwrite=True, max_retries=0)
        return
    execute_plan(*directories, overwrite=True, max_retries=0)


class TestTheCheckpointKeepsEverySimulatorsTail:
    """The storage half, on the checkpoint alone."""

    def test_a_later_simulator_does_not_overwrite_an_earlier_one_s_tail(self, tmp_path):
        """Simulator A completes with a tail, B completes after it, and A's tail must survive.

        With one tail per checkpoint the second save simply replaced the first, so A's spillover was
        unrecoverable -- and A is exactly the simulator a resume would need it for, since its batches
        run before B's.
        """
        manager = CheckpointManager(tmp_path)

        manager.save_checkpoint([0], "alpha", 0, {"counter": 1}, ["alpha-tail"])
        manager.save_checkpoint(
            [0, 1],
            "beta",
            1,
            {"counter": 1},
            ["beta-tail"],
            # `.get`, so a checkpoint that kept no per-simulator tails reaches the assertion below
            # rather than raising here.
            simulator_tails=(manager.load_checkpoint() or {}).get("simulator_tails"),
        )

        assert manager.get_last_simulator_spillover("alpha", 1) == ["alpha-tail"], (
            "the second simulator's checkpoint erased the first's spillover"
        )
        assert manager.get_last_simulator_state("alpha") == {"counter": 1}
        assert manager.get_last_simulator_spillover("beta", 2) == ["beta-tail"], "and the later one is still there"

    def test_a_checkpoint_from_before_per_simulator_tails_still_resumes(self, tmp_path):
        """An upgrade mid-run reads the single tail the old format wrote, rather than losing it."""
        (tmp_path / "simulation.checkpoint.json").write_text(
            json.dumps(
                {
                    "completed_batch_indices": [0],
                    "last_simulator_name": "alpha",
                    "last_completed_batch_index": 0,
                    "last_simulator_state": {"counter": 1},
                    "last_simulator_spillover": ["alpha-tail"],
                }
            )
        )
        manager = CheckpointManager(tmp_path)

        assert manager.get_last_simulator_spillover("alpha", 1) == ["alpha-tail"]
        assert manager.get_last_simulator_state("alpha") == {"counter": 1}
        assert manager.get_last_simulator_spillover("beta", 1) is None, "still not another simulator's"


class TestARunKeepsEverySimulatorsTail:
    """The wiring half: what ``execute_plan`` actually writes and reads back.

    The checkpoint tests above hand ``save_checkpoint`` the tails themselves, so a call site that
    never carries them forward leaves those green.
    """

    def test_the_final_checkpoint_still_holds_the_first_simulator_s_tail(self, tmp_path):
        """Run alpha to completion, then interrupt beta, and require alpha's tail in the file."""
        checkpoint_directory = tmp_path / "checkpoints"
        FAIL_AT_COUNTER["beta"] = 1  # beta's second batch, so beta has checkpointed once first

        _run(_plan(checkpoint_directory, {"alpha": [0, 1, 2], "beta": [3, 4]}), tmp_path, expect_failure=True)

        saved = json.loads((checkpoint_directory / "simulation.checkpoint.json").read_text())
        tails = saved.get("simulator_tails") or {}
        assert "alpha" in tails, "beta's checkpoint erased alpha's tail, which nothing can recover"
        assert tails["alpha"]["spillover"] == ["alpha-tail-2"], "alpha's last tail"
        assert tails["alpha"]["state"]["counter"] == 3, "and the state that goes with it"
        assert tails["beta"]["spillover"] == ["beta-tail-0"], "beta's own tail is kept too"

    def test_a_resumed_batch_is_not_handed_the_previous_simulator_s_state(self, tmp_path):
        """The state half, which fails *into* the data rather than out of it.

        alpha's counter after its three batches is 3, and beta's first batch is index 3, so the
        restore gate ``batch_index == state["counter"]`` matched -- and beta's first segment was
        generated from alpha's state: alpha's RNG stream, alpha's tag, in beta's data.
        """
        checkpoint_directory = tmp_path / "checkpoints"
        segments = {"alpha": [0, 1, 2], "beta": [3]}

        FAIL_AT_COUNTER["beta"] = 0
        _run(_plan(checkpoint_directory, segments), tmp_path, expect_failure=True)

        FAIL_AT_COUNTER.clear()
        _run(_plan(checkpoint_directory, segments), tmp_path, expect_failure=False)

        metadata = json.loads((tmp_path / "metadata" / "beta-3.metadata.json").read_text())
        assert metadata["pre_batch_state"]["tag"] == "beta", (
            "beta's segment was generated from alpha's state, so it carries alpha's RNG stream"
        )
        assert metadata["pre_batch_state"]["counter"] == 0, "beta's own first batch, not alpha's fourth"
        assert json.loads((tmp_path / "output" / "beta-3.json").read_text())["data"] == "beta-0"
