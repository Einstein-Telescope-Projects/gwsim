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

"""Which format `gwmock merge` writes, now that a run's own default is HDF5.

`--output` carries the choice, because gwmock selects a format by extension rather than by a flag. What
changed is the default: it used to be a hardcoded `merged.gwf`, and it now follows the inputs.

That distinction is the point of the item. HDF5 is the primary format for what gwmock *produces*, while
GWF exists so other pipelines can read the output -- so merging GWF frames for one of those pipelines
must not quietly hand back HDF5 just because HDF5 is what a run writes now.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gwmock.cli.merge import _resolve_output

pytestmark = pytest.mark.unit


class TestInferringTheFormat:
    """What happens when `--output` is not given."""

    def test_gwf_inputs_produce_a_gwf_merge(self) -> None:
        """The compatibility path: frames in, frames out."""
        assert _resolve_output(None, [Path("a.gwf"), Path("b.gwf")]) == "merged.gwf"

    def test_hdf5_inputs_produce_an_hdf5_merge(self) -> None:
        assert _resolve_output(None, [Path("a.hdf5"), Path("b.hdf5")]) == "merged.hdf5"

    def test_the_inferred_extension_is_the_inputs_own(self) -> None:
        """`.h5` is not silently normalised to `.hdf5`; the inputs' spelling is kept."""
        assert _resolve_output(None, [Path("a.h5"), Path("b.h5")]) == "merged.h5"

    def test_the_case_of_the_extension_does_not_split_the_inputs(self) -> None:
        """`A.GWF` and `b.gwf` are one format, not two, so this must not read as mixed."""
        assert _resolve_output(None, [Path("A.GWF"), Path("b.gwf")]) == "merged.gwf"


class TestWhenItRefuses:
    """The cases where guessing would discard what the caller asked for."""

    def test_mixed_inputs_raise_rather_than_choosing(self) -> None:
        """Merging a frame and an HDF5 file has two defensible answers, so it asks.

        Picking one would silently drop half the intent; naming both formats costs the caller one flag.
        """
        with pytest.raises(ValueError, match="mixture"):
            _resolve_output(None, [Path("a.gwf"), Path("b.hdf5")])

    def test_the_refusal_names_both_formats_and_the_way_out(self) -> None:
        """A refusal that does not say what to do next is a worse failure than a wrong guess."""
        with pytest.raises(ValueError, match="mixture") as raised:
            _resolve_output(None, [Path("a.gwf"), Path("b.hdf5")])
        message = str(raised.value)
        assert ".gwf" in message
        assert ".hdf5" in message
        assert "--output" in message

    def test_inputs_without_extensions_raise(self) -> None:
        """There is nothing to infer from, and defaulting to a format would be inventing one."""
        with pytest.raises(ValueError, match="extension"):
            _resolve_output(None, [Path("frame_a"), Path("frame_b")])


class TestWhenTheCallerChose:
    """`--output` decides, whatever the inputs are."""

    def test_an_explicit_output_is_returned_unchanged(self) -> None:
        assert _resolve_output("out.h5", [Path("a.gwf")]) == "out.h5"

    def test_an_explicit_output_overrides_a_mixture_instead_of_raising(self) -> None:
        """The mixture is only a problem because the format is unknown; naming it resolves that."""
        assert _resolve_output("combined.hdf5", [Path("a.gwf"), Path("b.hdf5")]) == "combined.hdf5"
