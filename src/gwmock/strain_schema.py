"""The declared, versioned schema of the strain artifacts gwmock writes.

A pipeline reading gwmock output needs to know what it is holding: which dataset carries the samples,
which attributes carry the grid, and -- when any of that changes -- that it changed. Until this module
existed the answer was only in gwmock's source, so a consumer had to match the writer's implementation
rather than a contract, and a layout change reached it as a wrong result rather than as a refusal.

What the schema declares is written into the artifact itself, at the **file root**:

``schema``
    ``"gwmock-strain"`` -- which contract this is, so a ``schema_version`` written by some other
    producer is not mistaken for this one.

``schema_version``
    ``MAJOR.MINOR.PATCH``. The major moves when a reader written against the previous version would
    misread a file (a dataset renamed, an attribute's meaning changed); the minor moves when something
    is added that such a reader can ignore.

Two consequences of that placement are deliberate:

* **Root, not dataset.** gwpy's HDF5 reader passes every dataset attribute to the series constructor as
  a keyword argument, so an attribute it does not know raises ``TypeError`` and the file becomes
  unreadable through the standard reader. Measured: adding ``schema_version`` to the dataset makes
  ``TimeSeries.read`` fail with ``Array.__new__() got an unexpected keyword argument 'schema_version'``;
  at the root it reads unchanged. The declaration also describes the file rather than one dataset in it,
  so a consumer can check it without first knowing the channel name.
* **Separate from the metadata record's version.** ``gwmock.cli.utils.metadata.SCHEMA_VERSION`` versions
  the *provenance record*, and its releases so far recorded changes to the meaning of
  ``signal.injections`` -- which say nothing about the layout of a strain file. Wiring one constant to
  both would announce a strain-format change on every metadata change and, worse, let a real strain
  change hide inside a metadata bump. They are two contracts and they move independently.

Only HDF5 carries the declaration, because only HDF5 has somewhere to put it: ``.npy`` is a bare array
container with no metadata space, and GWF frames are composed by the frame library from a fixed set of
fields. For those formats the run's metadata record remains the only description of what was written.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

#: The name written to the ``schema`` attribute, identifying which contract the file claims to meet.
STRAIN_SCHEMA = "gwmock-strain"

#: The version of the strain contract this gwmock writes.
#:
#: 1.0.0: one dataset per channel, named for the channel, holding the samples as float64 strain; the
#: grid in the dataset's ``x0`` (epoch, GPS seconds) and ``dx`` (sample interval, seconds) attributes,
#: with ``xunit``/``unit`` naming their units and ``channel``/``name`` repeating the channel. This is
#: the layout gwmock already wrote; 1.0.0 declares it rather than changing it.
STRAIN_SCHEMA_VERSION = "1.0.0"

#: Root attribute naming the schema.
SCHEMA_ATTRIBUTE = "schema"

#: Root attribute carrying the schema version.
SCHEMA_VERSION_ATTRIBUTE = "schema_version"

_HDF5_SUFFIXES = frozenset({".hdf5", ".h5"})
_VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


class StrainSchema(NamedTuple):
    """The schema declaration read from a strain artifact."""

    name: str
    version: str

    @property
    def major(self) -> int:
        """Return the major component of the declared version.

        Returns:
            The major version.
        """
        return parse_strain_schema_version(self.version)[0]


def parse_strain_schema_version(value: str) -> tuple[int, int, int]:
    """Split a declared version into its numeric components.

    Args:
        value: The version string.

    Returns:
        The major, minor and patch components.

    Raises:
        ValueError: If the version does not follow ``MAJOR.MINOR.PATCH``.
    """
    match = _VERSION_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"Strain schema version '{value}' must follow semantic versioning (MAJOR.MINOR.PATCH).")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def strain_schema_attributes() -> dict[str, str]:
    """Return the root attributes declaring the current strain schema.

    Returns:
        The attribute names and values to write at the root of a strain artifact.
    """
    return {SCHEMA_ATTRIBUTE: STRAIN_SCHEMA, SCHEMA_VERSION_ATTRIBUTE: STRAIN_SCHEMA_VERSION}


def carries_strain_schema(path: str | Path) -> bool:
    """Return whether a path names a format that can carry the declaration.

    Args:
        path: The artifact path.

    Returns:
        True for HDF5 artifacts, False for the formats with no attribute space.
    """
    return Path(path).suffix.lower() in _HDF5_SUFFIXES


def stamp_strain_schema(path: str | Path) -> bool:
    """Declare the current strain schema on an already-written artifact.

    Call it once per artifact, after the samples are written and before the file is published under its
    final name. It is a no-op for ``.npy`` and ``.gwf``, which have nowhere to record it, so a caller
    that writes several formats does not have to branch on the format itself.

    Args:
        path: The artifact to stamp.

    Returns:
        True if the declaration was written, False if the format cannot carry one.

    Raises:
        FileNotFoundError: If the artifact does not exist.
    """
    artifact = Path(path)
    if not carries_strain_schema(artifact):
        return False
    if not artifact.exists():
        raise FileNotFoundError(f"Cannot declare the strain schema on a file that does not exist: {artifact}")

    import h5py  # noqa: PLC0415  # deferred so importing the contract does not pull in the HDF5 stack

    with h5py.File(artifact, "a") as handle:
        handle.attrs.update(strain_schema_attributes())
    return True


def read_strain_schema(path: str | Path) -> StrainSchema | None:
    """Return the schema an artifact declares, or ``None`` if it declares none.

    ``None`` is the answer for a file written before the declaration existed, one written by another
    producer, and one in a format that cannot carry it -- three different situations that a consumer
    handles the same way: it has no contract and must fall back to the run's metadata record.

    Args:
        path: The artifact to read.

    Returns:
        The declared schema, or None.
    """
    artifact = Path(path)
    if not carries_strain_schema(artifact):
        return None

    import h5py  # noqa: PLC0415  # deferred so importing the contract does not pull in the HDF5 stack

    with h5py.File(artifact, "r") as handle:
        if SCHEMA_ATTRIBUTE not in handle.attrs or SCHEMA_VERSION_ATTRIBUTE not in handle.attrs:
            return None
        return StrainSchema(
            name=_as_text(handle.attrs[SCHEMA_ATTRIBUTE]),
            version=_as_text(handle.attrs[SCHEMA_VERSION_ATTRIBUTE]),
        )


def require_strain_schema(path: str | Path) -> StrainSchema:
    """Return the declared schema, refusing anything this gwmock cannot read.

    This is the consumer-side half of the contract: it turns a file whose layout may have moved under
    the reader into a refusal at open time, rather than a wrong number later.

    Args:
        path: The artifact to check.

    Returns:
        The declared schema.

    Raises:
        ValueError: If the artifact declares no schema, declares a different one, or declares a major
            version this gwmock does not know how to read.
    """
    declared = read_strain_schema(path)
    if declared is None:
        raise ValueError(
            f"{Path(path)} declares no strain schema. It predates the declaration or was written by "
            "another producer; read its run metadata to learn its layout."
        )
    if declared.name != STRAIN_SCHEMA:
        raise ValueError(f"{Path(path)} declares schema '{declared.name}', not '{STRAIN_SCHEMA}'.")
    current_major = parse_strain_schema_version(STRAIN_SCHEMA_VERSION)[0]
    if declared.major != current_major:
        raise ValueError(
            f"{Path(path)} declares strain schema major version {declared.major}; this gwmock reads "
            f"major version {current_major} ({STRAIN_SCHEMA_VERSION})."
        )
    return declared


def _as_text(value: object) -> str:
    """Decode an HDF5 attribute that may come back as bytes.

    Args:
        value: The raw attribute value.

    Returns:
        The attribute as text.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
