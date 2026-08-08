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

"""A resume must not mix two population catalogues into one run.

``run_fingerprint`` identified a run by its config *bytes* plus the resolved output and metadata
directories. A config names its population by **path**, so replacing that file's content leaves every
one of those inputs unchanged: the checkpoint is accepted, the batches it records are skipped, and the
run ends up holding some batches generated from the old catalogue and the rest from the new one.

**Measured before fixing**, with the CLI rather than this harness: a 3-batch run interrupted after the
first batch, the population CSV rewritten at the same path, the identical command resumed. Batch 0 kept
the old catalogue's event (detector-frame mass 30) while batches 1 and 2 carried the new one's (81, 82).
Exit code 0, no warning. The resume said so in its own log -- "Loaded checkpoint: 1 batches already",
"Skipping batch 0 (already completed from checkpoint)".

Hashing the referenced file closes it. The cost was measured rather than assumed: 18 ms for a 45 MB
one-million-row catalogue at ~2.5 GB/s, anchored against ``openssl speed -evp sha256`` reporting
2.75 GB/s on the same host. Against a run measured in minutes, that is free.

Scope is deliberately the population file alone -- the input this bug was found through -- not every
path a config can name.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from gwmock.cli import simulate_utils
from gwmock.cli.simulate_utils import execute_plan
from gwmock.cli.utils import checkpoint as checkpoint_utils
from gwmock.cli.utils.checkpoint import ForeignCheckpointError, run_fingerprint
from gwmock.cli.utils.config import GlobalsConfig, PopulationConfig, SimulatorConfig, SimulatorOutputConfig
from gwmock.cli.utils.simulation_plan import SimulationBatch, SimulationPlan

pytestmark = pytest.mark.unit

CATALOGUE_HEADER = "detector_frame_mass_1,detector_frame_mass_2,coa_time,distance\n"
OLD_CATALOGUE = CATALOGUE_HEADER + "30.0,25.0,1000000100.0,400.0\n"
NEW_CATALOGUE = CATALOGUE_HEADER + "80.0,75.0,1000000100.0,400.0\n"


def _plan(checkpoint_directory: Path, population_file: Path, batches: int = 3) -> SimulationPlan:
    """A plan whose config names *population_file*, as a real orchestration config does."""
    plan = SimulationPlan(checkpoint_directory=checkpoint_directory)
    for index in range(batches):
        plan.add_batch(
            SimulationBatch(
                simulator_name="mock",
                simulator_config=SimulatorConfig(
                    class_="tests.cli.test_cli_simulate.MockSimulator",
                    arguments={"seed": 42},
                    output=SimulatorOutputConfig(file_name=f"batch_{index}.json"),
                    population=PopulationConfig(
                        backend="FilePopulationLoader", arguments={"path": str(population_file)}
                    ),
                ),
                globals_config=GlobalsConfig(),
                batch_index=index,
            )
        )
    return plan


def test_swapping_the_population_behind_a_resume_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point: a resume across two catalogues must stop rather than mix them.

    The checkpoint is left behind by an **actually interrupted** run rather than hand-written, so the
    value on disk is whatever the production code computes. Hand-writing it with the new signature made
    this test fail on the unfixed code for the wrong reason -- a ``TypeError`` from an argument that did
    not exist yet -- which proves nothing about the defect.

    Fails on the unfixed code by *not raising*: the second run is waved through, batch 0 keeps the first
    catalogue's output, and the rest are generated from the second.
    """
    output_directory, metadata_directory = tmp_path / "output", tmp_path / "metadata"
    checkpoint_directory = tmp_path / "checkpoints"
    population = tmp_path / "population.csv"
    population.write_text(OLD_CATALOGUE)

    # Fail the last batch, so the run stops with a checkpoint on disk instead of cleaning it up.
    real_save = simulate_utils.save_metadata_record

    def _fail_on_the_last_batch(*args, **kwargs):
        record = args[0] if args else kwargs.get("record")
        if "mock-2" in str(kwargs.get("metadata_file", "")) or "mock-2" in str(record):
            raise RuntimeError("interrupted")
        return real_save(*args, **kwargs)

    monkeypatch.setattr(simulate_utils, "save_metadata_record", _fail_on_the_last_batch)
    with pytest.raises(RuntimeError, match="interrupted"):
        execute_plan(_plan(checkpoint_directory, population), output_directory, metadata_directory, overwrite=True)
    monkeypatch.undo()

    saved = json.loads((checkpoint_directory / "simulation.checkpoint.json").read_text())
    assert saved.get("completed_batch_indices"), "the interrupted run left no completed batches to skip"

    population.write_text(NEW_CATALOGUE)  # same path, same config bytes, different catalogue

    with pytest.raises(ForeignCheckpointError):
        execute_plan(_plan(checkpoint_directory, population), output_directory, metadata_directory, overwrite=True)


class TestTheFingerprintItself:
    """The identity function, at the level a future change is most likely to break it."""

    def test_the_population_content_changes_it(self, tmp_path: Path) -> None:
        """Same path, same config bytes, different bytes behind it: a different run."""
        population = tmp_path / "population.csv"
        population.write_text(OLD_CATALOGUE)
        before = run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta", [population])
        population.write_text(NEW_CATALOGUE)
        after = run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta", [population])
        assert before != after, "the catalogue behind the config was replaced and the run looked identical"

    def test_identical_content_at_a_different_path_still_differs(self, tmp_path: Path) -> None:
        """Content is added to the identity, not substituted for the path.

        Two catalogues with the same bytes are still two inputs: the path is part of the config, and a
        run that reads a different file is a different run even when today's bytes agree. Substituting
        content for path would also make a *renamed* input invisible, which is caught today.
        """
        first, second = tmp_path / "a.csv", tmp_path / "b.csv"
        first.write_text(OLD_CATALOGUE)
        second.write_text(OLD_CATALOGUE)
        assert run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta", [first]) != run_fingerprint(
            ["a" * 64], tmp_path / "out", tmp_path / "meta", [second]
        )

    def test_a_remote_input_is_never_read_from_disk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A URL must not reach the filesystem, which is what "kept as its URL" has to mean.

        Comparing two URLs' fingerprints does not show this: with the remote branch removed, a URL falls
        through to a failed ``open`` and gets a marker, and because the reference string is part of the
        key the two fingerprints still differ. The assertion has to be that no file access happens --
        that mutation survived every other test here.
        """

        def _explode(*args, **kwargs):
            raise AssertionError("a remote reference was opened as a local path")

        monkeypatch.setattr(Path, "open", _explode)
        assert checkpoint_utils._input_digest("https://example.invalid/bbh_population.csv") == "<remote>"

    def test_a_missing_input_does_not_crash_the_run(self, tmp_path: Path) -> None:
        """An unreadable or absent input is a marker, not an exception.

        The fingerprint is computed before the run reads anything, and a population that does not exist
        yet -- a remote URL, a path typo, a file staged later -- must reach its own error rather than a
        traceback from the identity check. It still has to change the fingerprint, so a run that could
        not hash its input is not mistaken for one that did.

        The marker's *bracketed shape* is load-bearing since the unverified-input warning was added --
        ``_is_marker`` distinguishes "gave up" from "hashed" by it -- so a mutation that returns some
        other constant now silences that warning rather than being harmless. That is pinned by
        ``test_a_resume_says_it_could_not_verify_an_unreadable_population`` below. The exact spelling
        still is not asserted, only that it is not mistakable for a digest.
        """
        missing = tmp_path / "not-there.csv"
        unhashed = run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta", [missing])
        present = tmp_path / "population.csv"
        present.write_text(OLD_CATALOGUE)
        hashed = run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta", [present])
        assert unhashed != hashed
        assert unhashed != run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta", [])

    def test_a_remote_input_is_kept_as_its_url(self, tmp_path: Path) -> None:
        """A URL cannot be hashed without fetching it, and this must not try.

        The examples pin their population to an `https://` URL. Fetching here would download the
        catalogue a second time purely to identify the run, so a remote input contributes its URL --
        which the config bytes already cover -- and the gap is documented rather than papered over.
        """
        url = "https://example.invalid/bbh_population.csv"
        assert run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta", [url]) == run_fingerprint(
            ["a" * 64], tmp_path / "out", tmp_path / "meta", [url]
        )
        assert run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta", [url]) != run_fingerprint(
            ["a" * 64], tmp_path / "out", tmp_path / "meta", ["https://example.invalid/other.csv"]
        )

    def test_the_order_of_inputs_does_not_invent_a_new_run(self, tmp_path: Path) -> None:
        """Two batches naming the same file must not fingerprint differently from one that does.

        Every batch carries the population config, so a plan of three batches presents the same path
        three times. Left unnormalised, adding a batch would change the input part of the identity for a
        reason that has nothing to do with the inputs.
        """
        population = tmp_path / "population.csv"
        population.write_text(OLD_CATALOGUE)
        once = run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta", [population])
        thrice = run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta", [population] * 3)
        assert once == thrice


def test_the_recorded_fingerprint_is_what_the_next_run_compares(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The checkpoint stores the fingerprint that includes the input, not the bare config hash.

    Without this the fix is inert in a way no other test here would catch: the identity could cover the
    catalogue while the value written to disk, and therefore compared on resume, does not.

    The run is interrupted rather than completed, because a successful run deletes its checkpoint -- the
    first version of this test read a file that had just been cleaned up.
    """
    output_directory, metadata_directory = tmp_path / "output", tmp_path / "metadata"
    checkpoint_directory = tmp_path / "checkpoints"
    population = tmp_path / "population.csv"
    population.write_text(OLD_CATALOGUE)
    plan = _plan(checkpoint_directory, population)

    real_save = simulate_utils.save_metadata_record

    def _fail_on_the_last_batch(*args, **kwargs):
        if "mock-2" in str(kwargs.get("metadata_file", "")) or "mock-2" in str(args[0] if args else ""):
            raise RuntimeError("interrupted")
        return real_save(*args, **kwargs)

    monkeypatch.setattr(simulate_utils, "save_metadata_record", _fail_on_the_last_batch)
    with pytest.raises(RuntimeError, match="interrupted"):
        execute_plan(plan, output_directory, metadata_directory, overwrite=True)
    monkeypatch.undo()

    saved = json.loads((checkpoint_directory / "simulation.checkpoint.json").read_text())
    with_the_input = run_fingerprint(
        [batch.config_sha256 for batch in plan.batches], output_directory, metadata_directory, [population]
    )
    without_it = run_fingerprint([batch.config_sha256 for batch in plan.batches], output_directory, metadata_directory)
    assert with_the_input != without_it, "the two fingerprints agree, so this test cannot discriminate"
    assert saved.get("config_sha256") == with_the_input


def test_every_batch_s_population_counts_not_only_the_first(tmp_path: Path) -> None:
    """A plan can name more than one catalogue, and each one is part of the run's identity.

    Every batch in the earlier tests names the same file, so a change that hashed only the first batch's
    population passed all of them. A plan assembled from several metadata records is the real case --
    the same reasoning the config-hash side already documents.
    """
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    first.write_text(OLD_CATALOGUE)
    second.write_text(OLD_CATALOGUE)

    before = run_fingerprint(["a" * 64, "b" * 64], tmp_path / "out", tmp_path / "meta", [first, second])
    second.write_text(NEW_CATALOGUE)  # the *second* batch's catalogue changes
    after = run_fingerprint(["a" * 64, "b" * 64], tmp_path / "out", tmp_path / "meta", [first, second])

    assert before != after, "a later batch's catalogue was replaced and the run looked identical"


def test_the_plan_s_populations_are_collected_from_every_batch(tmp_path: Path) -> None:
    """The collector itself, since the fingerprint cannot see a path the collector dropped."""
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    first.write_text(OLD_CATALOGUE)
    second.write_text(NEW_CATALOGUE)
    plan = _plan(tmp_path / "checkpoints", first, batches=1)
    plan.add_batch(
        SimulationBatch(
            simulator_name="mock",
            simulator_config=SimulatorConfig(
                class_="tests.cli.test_cli_simulate.MockSimulator",
                arguments={"seed": 42},
                output=SimulatorOutputConfig(file_name="batch_1.json"),
                population=PopulationConfig(backend="FilePopulationLoader", arguments={"path": str(second)}),
            ),
            globals_config=GlobalsConfig(),
            batch_index=1,
        )
    )

    collected = simulate_utils._referenced_population_files(plan)

    assert collected == [str(first), str(second)], f"a batch's population was dropped: {collected}"


class TestWhatTheLoaderActuallyReads:
    """Findings from review: the identity has to follow the loader, not a plausible guess at it."""

    def test_a_tilde_path_is_expanded_before_hashing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The live false negative: a hand-written `~/...` path silently verified nothing.

        The loader expands `~` before reading. This did not, so the open failed, the same marker was
        recorded whatever the file held, and the guard did nothing for exactly the paths people type.
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        population = tmp_path / "population.csv"
        population.write_text(OLD_CATALOGUE)

        before = run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta", ["~/population.csv"])
        population.write_text(NEW_CATALOGUE)
        after = run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta", ["~/population.csv"])

        assert before != after, "a `~` path was never opened, so the catalogue behind it was never checked"

    def test_the_remote_predicate_agrees_with_the_loader(self) -> None:
        """Drift between our scheme test and the loader's is the defect, so compare them directly.

        Skipped rather than replicated if the loader's helper moves: this asserts agreement, and an
        assertion against an import that no longer exists would fail for the wrong reason.
        """
        loader_predicate = pytest.importorskip("gwmock_pop.loaders._fetch").is_population_url
        for reference in (
            "https://example.invalid/pop.csv",
            "http://example.invalid/pop.csv",
            "s3://bucket/pop.csv",
            "zenodo://12345/pop.csv",
            "data://pop.csv",  # a directory literally named `data:` -- local to the loader
            "file:///tmp/pop.csv",
            "ftp://example.invalid/pop.csv",
            "/absolute/pop.csv",
            "relative/pop.csv",
            "~/pop.csv",
        ):
            assert checkpoint_utils._is_remote(reference) is bool(loader_predicate(reference)), (
                f"the identity and the loader disagree about {reference!r}"
            )

    def test_a_regenerated_catalogue_with_different_line_endings_still_resumes(self, tmp_path: Path) -> None:
        """The run consumes the parsed catalogue, so byte-identical parses must not refuse a resume.

        Verified in review: a trailing newline, a blank line, or CRLF/LF each changed the fingerprint while
        the parsed catalogue was identical. Population files are machine-generated, so a same-content
        regeneration is a normal workflow -- and refusing it costs a long run for nothing.
        """
        population = tmp_path / "population.csv"
        population.write_text(OLD_CATALOGUE)
        original = run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta", [population])

        population.write_bytes(OLD_CATALOGUE.replace("\n", "\r\n").encode())
        assert run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta", [population]) == original, (
            "the same catalogue rewritten with CRLF line endings refused the resume"
        )

    def test_a_reordered_catalogue_still_refuses(self, tmp_path: Path) -> None:
        """The normalisation is line endings only, and must not become "parse it and compare".

        A reordered or reformatted catalogue is a different input as far as this guard is concerned --
        the safe direction, and the boundary the previous test could otherwise be widened past.
        """
        population = tmp_path / "population.csv"
        population.write_text(CATALOGUE_HEADER + "30.0,25.0,1000000100.0,400.0\n80.0,75.0,1000000200.0,400.0\n")
        original = run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta", [population])
        population.write_text(CATALOGUE_HEADER + "80.0,75.0,1000000200.0,400.0\n30.0,25.0,1000000100.0,400.0\n")
        assert run_fingerprint(["a" * 64], tmp_path / "out", tmp_path / "meta", [population]) != original

    def test_a_resume_says_which_populations_it_could_not_verify(self, caplog: pytest.LogCaptureFixture) -> None:
        """A remote population gets no content coverage, and the operator has to be told.

        The marker adds nothing the config hash already carried, so for a remote catalogue this guard
        cannot refuse a mixed resume at all. Warning is the honest alternative to a docstring nobody reads.
        """
        with caplog.at_level(logging.WARNING, logger="gwmock"):
            checkpoint_utils.warn_if_inputs_are_unverified(["https://example.invalid/pop.csv"], resuming=True)
        assert [r for r in caplog.records if "could not verify" in r.message], "a remote resume said nothing"

    def test_a_resume_says_it_could_not_verify_an_unreadable_population(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An unreadable catalogue is as unverified as a remote one, and warns the same way.

        This is what makes the marker's shape matter rather than cosmetic: a change that returned some
        other constant for the unreadable branch would leave the fingerprint working and the warning
        silently off, which no other test here would notice.
        """
        with caplog.at_level(logging.WARNING, logger="gwmock"):
            checkpoint_utils.warn_if_inputs_are_unverified([tmp_path / "never-staged.csv"], resuming=True)
        assert [r for r in caplog.records if "could not verify" in r.message], (
            "a resume over a population it could not read said nothing"
        )

    def test_a_clean_first_run_says_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        """Not on a first run, and not when everything is verifiable: warnings that always fire are noise."""
        with caplog.at_level(logging.WARNING, logger="gwmock"):
            checkpoint_utils.warn_if_inputs_are_unverified(["https://example.invalid/pop.csv"], resuming=False)
        assert not [r for r in caplog.records if "could not verify" in r.message]


def test_a_real_resume_over_a_remote_population_emits_the_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The warning has to be reached by the resume, not merely exist.

    Every other test of it calls the function directly, so removing its one call site left them all
    passing -- the same way the fingerprint change itself could have been made inert. This drives an
    actual interrupted run and resume, with a population the identity cannot verify.
    """
    output_directory, metadata_directory = tmp_path / "output", tmp_path / "metadata"
    checkpoint_directory = tmp_path / "checkpoints"
    remote = "https://example.invalid/bbh_population.csv"

    def _plan_with_remote() -> SimulationPlan:
        plan = SimulationPlan(checkpoint_directory=checkpoint_directory)
        for index in range(3):
            plan.add_batch(
                SimulationBatch(
                    simulator_name="mock",
                    simulator_config=SimulatorConfig(
                        class_="tests.cli.test_cli_simulate.MockSimulator",
                        arguments={"seed": 42},
                        output=SimulatorOutputConfig(file_name=f"batch_{index}.json"),
                        population=PopulationConfig(backend="FilePopulationLoader", arguments={"path": remote}),
                    ),
                    globals_config=GlobalsConfig(),
                    batch_index=index,
                )
            )
        return plan

    real_save = simulate_utils.save_metadata_record

    def _fail_on_the_last_batch(*args, **kwargs):
        if "mock-2" in str(kwargs.get("metadata_file", "")) or "mock-2" in str(args[0] if args else ""):
            raise RuntimeError("interrupted")
        return real_save(*args, **kwargs)

    monkeypatch.setattr(simulate_utils, "save_metadata_record", _fail_on_the_last_batch)
    with pytest.raises(RuntimeError, match="interrupted"):
        execute_plan(_plan_with_remote(), output_directory, metadata_directory, overwrite=True)
    monkeypatch.undo()

    with caplog.at_level(logging.WARNING, logger="gwmock"):
        execute_plan(_plan_with_remote(), output_directory, metadata_directory, overwrite=True)

    unverified = [r for r in caplog.records if "could not verify" in r.message]
    assert unverified, "a resume over a remote population said nothing about the gap"
    assert remote in unverified[0].message
