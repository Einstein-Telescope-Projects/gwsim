"""The declared, versioned strain schema: what gwmock stamps on a strain file, and what it refuses.

Before this contract existed a pipeline reading gwmock output matched the writer's implementation --
which dataset it happened to create, which attributes it happened to set -- and a change to that layout
reached the reader as a wrong number rather than as a refusal. These tests pin the two halves: the
declaration a writer puts on the artifact, and the check a consumer runs before trusting it.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from gwmock.strain_schema import (
    SCHEMA_ATTRIBUTE,
    SCHEMA_VERSION_ATTRIBUTE,
    STRAIN_SCHEMA,
    STRAIN_SCHEMA_VERSION,
    StrainSchema,
    carries_strain_schema,
    parse_strain_schema_version,
    read_strain_schema,
    require_strain_schema,
    stamp_strain_schema,
    strain_schema_attributes,
)

pytestmark = pytest.mark.unit

RATE = 8.0
START = 100.0


def _hdf5_strain(path: Path, *, channel: str = "H1:MOCK_NOISE") -> Path:
    """Write a strain file laid out the way gwmock lays one out, with nothing declared on it yet."""
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(channel, data=np.arange(16, dtype=float))
        dataset.attrs["x0"] = START
        dataset.attrs["dx"] = 1.0 / RATE
        dataset.attrs["xunit"] = "s"
        dataset.attrs["channel"] = channel
        dataset.attrs["name"] = channel
        dataset.attrs["unit"] = "strain"
    return path


class TestWhatTheDeclarationSays:
    """The constants are the contract, so their shape is part of it."""

    def test_the_version_is_a_semantic_version(self) -> None:
        assert parse_strain_schema_version(STRAIN_SCHEMA_VERSION) == (1, 0, 0)

    def test_the_attributes_name_the_schema_and_its_version(self) -> None:
        assert strain_schema_attributes() == {
            SCHEMA_ATTRIBUTE: STRAIN_SCHEMA,
            SCHEMA_VERSION_ATTRIBUTE: STRAIN_SCHEMA_VERSION,
        }

    @pytest.mark.parametrize("value", ["1", "1.0", "1.0.0.0", "v1.0.0", "1.0.0-rc1", "01.0.0", "", "one.0.0"])
    def test_a_version_that_is_not_one_is_refused(self, value: str) -> None:
        """A reader compares majors, so a version it cannot split is not a version it can compare."""
        with pytest.raises(ValueError, match="semantic versioning"):
            parse_strain_schema_version(value)


class TestStamping:
    """What `stamp_strain_schema` writes, and where."""

    def test_an_hdf5_artifact_is_stamped(self, tmp_path: Path) -> None:
        path = _hdf5_strain(tmp_path / "strain.hdf5")

        assert stamp_strain_schema(path) is True
        assert read_strain_schema(path) == StrainSchema(name=STRAIN_SCHEMA, version=STRAIN_SCHEMA_VERSION)

    def test_the_h5_spelling_is_the_same_format(self, tmp_path: Path) -> None:
        path = _hdf5_strain(tmp_path / "strain.h5")

        assert stamp_strain_schema(path) is True
        assert read_strain_schema(path) is not None

    def test_the_declaration_is_written_at_the_root(self, tmp_path: Path) -> None:
        """Where it goes is the whole design, not a detail -- see the sibling test below for why."""
        path = _hdf5_strain(tmp_path / "strain.hdf5")

        stamp_strain_schema(path)

        with h5py.File(path, "r") as handle:
            assert handle.attrs[SCHEMA_ATTRIBUTE] == STRAIN_SCHEMA
            assert handle.attrs[SCHEMA_VERSION_ATTRIBUTE] == STRAIN_SCHEMA_VERSION
            assert SCHEMA_VERSION_ATTRIBUTE not in handle["H1:MOCK_NOISE"].attrs

    def test_a_stamped_file_still_reads_through_gwpy(self, tmp_path: Path) -> None:
        """The reason the declaration is not a dataset attribute.

        gwpy hands every dataset attribute to the series constructor as a keyword argument, so an
        attribute it does not know raises `TypeError: Array.__new__() got an unexpected keyword argument
        'schema_version'` and the file becomes unreadable through the standard reader. Declaring the
        schema must not cost a consumer the ability to open the file.
        """
        path = _hdf5_strain(tmp_path / "strain.hdf5")

        stamp_strain_schema(path)

        series = TimeSeries.read(str(path), format="hdf5")
        assert series.t0.value == START
        assert series.sample_rate.value == RATE
        assert np.array_equal(series.value, np.arange(16, dtype=float))

    def test_the_samples_are_untouched(self, tmp_path: Path) -> None:
        path = _hdf5_strain(tmp_path / "strain.hdf5")

        stamp_strain_schema(path)

        with h5py.File(path, "r") as handle:
            dataset = handle["H1:MOCK_NOISE"]
            assert np.array_equal(dataset[()], np.arange(16, dtype=float))
            assert dict(dataset.attrs)["x0"] == START

    def test_stamping_twice_leaves_one_declaration(self, tmp_path: Path) -> None:
        """A file may be re-stamped -- a merge of merges, a re-run over an existing path."""
        path = _hdf5_strain(tmp_path / "strain.hdf5")

        stamp_strain_schema(path)
        stamp_strain_schema(path)

        with h5py.File(path, "r") as handle:
            assert sorted(handle.attrs) == sorted([SCHEMA_ATTRIBUTE, SCHEMA_VERSION_ATTRIBUTE])

    @pytest.mark.parametrize("name", ["strain.npy", "strain.gwf", "strain.txt"])
    def test_a_format_with_no_attribute_space_is_skipped(self, tmp_path: Path, name: str) -> None:
        """`.npy` is a bare array container and a GWF frame is composed from a fixed set of fields, so
        neither can carry the declaration. A caller writing several formats says so once, here, rather
        than branching on the format at every write site."""
        path = tmp_path / name
        path.write_bytes(b"not hdf5")

        assert stamp_strain_schema(path) is False
        assert path.read_bytes() == b"not hdf5"

    def test_a_missing_hdf5_artifact_is_an_error(self, tmp_path: Path) -> None:
        """It is called straight after a write, so an absent file means the write did not happen."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            stamp_strain_schema(tmp_path / "never-written.hdf5")

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("a.hdf5", True), ("a.h5", True), ("A.HDF5", True), ("a.gwf", False), ("a.npy", False), ("a", False)],
    )
    def test_which_formats_can_carry_it(self, name: str, expected: bool) -> None:
        assert carries_strain_schema(Path(name)) is expected


class TestReadingTheDeclaration:
    """What a consumer sees when it looks."""

    def test_an_undeclared_file_reads_as_none(self, tmp_path: Path) -> None:
        """A file written before the declaration existed, or by another producer."""
        assert read_strain_schema(_hdf5_strain(tmp_path / "strain.hdf5")) is None

    def test_a_half_declared_file_reads_as_none(self, tmp_path: Path) -> None:
        """A version with no schema name says nothing: `schema_version` is a common attribute name and
        another producer's is not this contract."""
        path = _hdf5_strain(tmp_path / "strain.hdf5")
        with h5py.File(path, "a") as handle:
            handle.attrs[SCHEMA_VERSION_ATTRIBUTE] = "1.0.0"

        assert read_strain_schema(path) is None

    def test_a_format_that_cannot_carry_it_reads_as_none(self, tmp_path: Path) -> None:
        path = tmp_path / "strain.npy"
        np.save(path, np.arange(4, dtype=float))

        assert read_strain_schema(path) is None

    def test_a_byte_string_attribute_is_decoded(self, tmp_path: Path) -> None:
        """h5py hands back `bytes` for an attribute written by a writer that used a fixed-length string
        type, which is what a non-Python producer typically writes."""
        path = _hdf5_strain(tmp_path / "strain.hdf5")
        with h5py.File(path, "a") as handle:
            handle.attrs.create(SCHEMA_ATTRIBUTE, np.bytes_(STRAIN_SCHEMA))
            handle.attrs.create(SCHEMA_VERSION_ATTRIBUTE, np.bytes_(STRAIN_SCHEMA_VERSION))

        assert read_strain_schema(path) == StrainSchema(name=STRAIN_SCHEMA, version=STRAIN_SCHEMA_VERSION)


class TestRequiringTheDeclaration:
    """The consumer-side half: what gets refused, and with what said about it."""

    def test_a_stamped_file_is_accepted(self, tmp_path: Path) -> None:
        path = _hdf5_strain(tmp_path / "strain.hdf5")
        stamp_strain_schema(path)

        assert require_strain_schema(path).version == STRAIN_SCHEMA_VERSION

    def test_an_undeclared_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="declares no strain schema"):
            require_strain_schema(_hdf5_strain(tmp_path / "strain.hdf5"))

    def test_another_producers_schema_is_refused(self, tmp_path: Path) -> None:
        path = _hdf5_strain(tmp_path / "strain.hdf5")
        with h5py.File(path, "a") as handle:
            handle.attrs[SCHEMA_ATTRIBUTE] = "someone-elses-strain"
            handle.attrs[SCHEMA_VERSION_ATTRIBUTE] = "1.0.0"

        with pytest.raises(ValueError, match="declares schema 'someone-elses-strain'"):
            require_strain_schema(path)

    def test_a_future_major_version_is_refused(self, tmp_path: Path) -> None:
        """The point of the version: a layout this reader was not written against is a refusal at open
        time rather than a wrong number later."""
        path = _hdf5_strain(tmp_path / "strain.hdf5")
        with h5py.File(path, "a") as handle:
            handle.attrs[SCHEMA_ATTRIBUTE] = STRAIN_SCHEMA
            handle.attrs[SCHEMA_VERSION_ATTRIBUTE] = "2.0.0"

        with pytest.raises(ValueError, match="major version 2"):
            require_strain_schema(path)

    def test_a_later_minor_version_is_accepted(self, tmp_path: Path) -> None:
        """A minor release adds what an existing reader can ignore, so refusing it would be wrong."""
        path = _hdf5_strain(tmp_path / "strain.hdf5")
        with h5py.File(path, "a") as handle:
            handle.attrs[SCHEMA_ATTRIBUTE] = STRAIN_SCHEMA
            handle.attrs[SCHEMA_VERSION_ATTRIBUTE] = "1.7.3"

        assert require_strain_schema(path).version == "1.7.3"

    def test_a_malformed_version_is_refused(self, tmp_path: Path) -> None:
        path = _hdf5_strain(tmp_path / "strain.hdf5")
        with h5py.File(path, "a") as handle:
            handle.attrs[SCHEMA_ATTRIBUTE] = STRAIN_SCHEMA
            handle.attrs[SCHEMA_VERSION_ATTRIBUTE] = "not-a-version"

        with pytest.raises(ValueError, match="semantic versioning"):
            require_strain_schema(path)
