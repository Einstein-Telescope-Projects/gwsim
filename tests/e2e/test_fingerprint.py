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

"""What the reference fingerprint distinguishes, and what it therefore stops misreporting.

These are plain functions, so they are unit tests rather than part of the ``e2e`` matrix: the matrix
needs a full generation run, and none of that is required to check which environments compare equal.

The behaviour is worth pinning because the failure it prevents is silent. A CPU-written reference
replayed on a GPU produces different bytes -- measured on one host, all three output frames differ -- and
without the device in the fingerprint the two environments compared equal, so the bit-mismatch note
blamed "references written somewhere subtly different" for what was simply the other device.
"""

from __future__ import annotations

import pytest

from .reference_values import fingerprint, same_environment

pytestmark = pytest.mark.unit

LINUX_CPU = {"system": "Linux", "machine": "x86_64", "python": "3.12", "device": "cpu"}


class TestSameEnvironment:
    """The comparison the bit-mismatch note depends on."""

    def test_a_run_matches_itself(self) -> None:
        assert same_environment(LINUX_CPU, dict(LINUX_CPU))

    def test_a_cpu_reference_does_not_match_a_gpu_run(self) -> None:
        """The case the device key exists for.

        Both reviewers found this returning ``True`` when the comparison used only the stored record's
        keys, which left the key inert for every reference in the tree and kept the misleading note.
        """
        assert not same_environment(LINUX_CPU, dict(LINUX_CPU, device="gpu"))

    def test_an_unlabelled_reference_does_not_match_a_labelled_run(self) -> None:
        """A reference predating the device key is not silently treated as matching.

        Suppressing the note for such a reference is the intended cost: a note that misattributes is
        worse than one that is missing, and every reference in the tree now carries the key anyway.
        """
        unlabelled = {key: value for key, value in LINUX_CPU.items() if key != "device"}
        assert not same_environment(unlabelled, LINUX_CPU)

    @pytest.mark.parametrize("key", ["system", "machine", "python", "device"])
    def test_every_recorded_key_can_separate_two_environments(self, key: str) -> None:
        """No key is decorative: each one alone makes the environments differ.

        Without this a key could be added to the record and never consulted -- which is exactly what
        happened to ``device`` before the comparison was made symmetric.
        """
        assert not same_environment(LINUX_CPU, dict(LINUX_CPU, **{key: "something-else"}))

    def test_a_reference_without_a_fingerprint_matches_nothing(self) -> None:
        """There is nothing to compare, so the honest answer is "not the same environment"."""
        assert not same_environment(None, LINUX_CPU)
        assert not same_environment({}, LINUX_CPU)
        assert not same_environment(LINUX_CPU, None)


class TestFingerprint:
    """What a run records about itself."""

    def test_it_records_the_device(self) -> None:
        assert "device" in fingerprint()

    def test_the_device_is_one_of_the_states_the_docstring_names(self) -> None:
        """`none` and `unavailable` are real states, not defensive decoration.

        `unavailable` was written for a CUDA plugin that raises rather than falling back, and a real
        host produced exactly that: CUDA jaxlib installed, with cards below the compute capability
        current JAX supports.
        """
        device = fingerprint()["device"]
        assert device in {"cpu", "gpu", "tpu", "none", "unavailable"}, device

    def test_it_still_records_what_it_did_before(self) -> None:
        """The device is an addition; dropping a key would silently widen what compares equal."""
        assert set(fingerprint()) >= {"system", "machine", "python"}
