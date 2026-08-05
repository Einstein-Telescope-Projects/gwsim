"""Tests for the gwmock population adapter."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from gwmock_pop import CBC_PARAMETER_NAMES, ExternalPopulationLoader, GWPopSimulator

from gwmock.cli.utils.backend_resolver import instantiate_backend, resolve_backend_class
from gwmock.population import PopulationAdapter

EXPECTED_SAMPLE_COUNT = 2


class MockGWPopBackend:
    """Protocol-compatible simulator backend for adapter tests."""

    parameter_names = (
        "detector_frame_mass_1",
        "detector_frame_mass_2",
        "redshift",
        "coa_time",
    )
    source_type = "bbh"

    def simulate(self, n_samples: int, **kwargs):
        if n_samples != EXPECTED_SAMPLE_COUNT:
            raise AssertionError("Unexpected n_samples for test backend.")
        return {
            "detector_frame_mass_1": np.array([30.0, 32.0]),
            "detector_frame_mass_2": np.array([20.0, 21.0]),
            "redshift": np.array([0.1, 0.2]),
            "coa_time": np.array([1000.0, 1001.0]),
        }


class ModulePopulationBackend(MockGWPopBackend):
    """Protocol-compatible backend for module:Class resolver tests."""


class MockExternalPopulationLoader:
    """Protocol-compatible file loader backend for adapter tests."""

    parameter_names: ClassVar[tuple[str, ...]] = (
        "detector_frame_mass_1",
        "detector_frame_mass_2",
        "inclination",
        "coa_time",
    )
    source_type = "bns"
    metadata: ClassVar[dict[str, str]] = {"resolved_path": str(Path(tempfile.gettempdir()) / "catalog.h5")}

    def simulate(self, n_samples: int, **kwargs):
        if n_samples != EXPECTED_SAMPLE_COUNT:
            raise AssertionError("Unexpected n_samples for test loader.")
        return {
            "detector_frame_mass_1": np.array([1.4, 1.35]),
            "detector_frame_mass_2": np.array([1.3, 1.25]),
            "inclination": np.array([0.2, 0.3]),
            "coa_time": np.array([2000.0, 2001.0]),
        }


def _install_entry_point_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    alias: str,
    package_name: str,
) -> None:
    package_dir = tmp_path / "plugin-src"
    site_packages = tmp_path / "site-packages"
    (package_dir / package_name).mkdir(parents=True)
    (package_dir / package_name / "__init__.py").write_text("")
    (package_dir / package_name / "population.py").write_text(
        textwrap.dedent(
            """
            import numpy as np

            class EntryPointPopulationBackend:
                parameter_names = ("detector_frame_mass_1",)
                source_type = "bbh"

                def simulate(self, n_samples: int, **_kwargs):
                    return {"detector_frame_mass_1": np.full(n_samples, 7.0)}
            """
        )
    )
    (package_dir / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [build-system]
            requires = ["setuptools>=61"]
            build-backend = "setuptools.build_meta"

            [project]
            name = "{package_name}"
            version = "0.0.1"

            [project.entry-points."gwmock.population"]
            """
        ).format(package_name=package_name)
        + f'{alias} = "{package_name}.population:EntryPointPopulationBackend"\n'
    )

    uv_path = shutil.which("uv")
    if uv_path is None:  # pragma: no cover - repository tests run with uv available
        raise AssertionError("uv executable is required for entry-point installation tests.")

    subprocess.run(  # noqa: S603
        [
            uv_path,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--quiet",
            "--no-deps",
            "--target",
            str(site_packages),
            str(package_dir),
        ],
        check=True,
    )
    monkeypatch.syspath_prepend(str(site_packages))


class TestPopulationAdapter:
    """Test suite for population adapter behavior."""

    def test_from_backend_accepts_gwpop_simulator(self):
        """Simulator-backed batches are sliced into per-event dictionaries."""
        backend = instantiate_backend("population", "tests.population.test_adapter:MockGWPopBackend")

        assert isinstance(backend, GWPopSimulator)

        adapter = PopulationAdapter.from_backend(backend, n_samples=EXPECTED_SAMPLE_COUNT)

        events = list(adapter)

        assert len(adapter) == EXPECTED_SAMPLE_COUNT
        assert adapter.source_type == "bbh"
        assert adapter.parameter_names == backend.parameter_names
        assert list(events[0].keys()) == list(backend.parameter_names)
        assert set(adapter.parameter_names).issubset(CBC_PARAMETER_NAMES)
        assert events == [
            {
                "detector_frame_mass_1": 30.0,
                "detector_frame_mass_2": 20.0,
                "redshift": 0.1,
                "coa_time": 1000.0,
            },
            {
                "detector_frame_mass_1": 32.0,
                "detector_frame_mass_2": 21.0,
                "redshift": 0.2,
                "coa_time": 1001.0,
            },
        ]

    def test_from_backend_accepts_external_population_loader(self):
        """Loader-backed batches use the same adapter boundary."""
        loader = instantiate_backend(
            "population",
            "tests.population.test_adapter:MockExternalPopulationLoader",
        )

        assert isinstance(loader, GWPopSimulator)
        assert isinstance(loader, ExternalPopulationLoader)

        adapter = PopulationAdapter.from_backend(loader, n_samples=EXPECTED_SAMPLE_COUNT)

        assert adapter.source_type == "bns"
        assert adapter.metadata == loader.metadata
        assert list(adapter.iter_event_parameters()) == [
            {
                "detector_frame_mass_1": 1.4,
                "detector_frame_mass_2": 1.3,
                "inclination": 0.2,
                "coa_time": 2000.0,
            },
            {
                "detector_frame_mass_1": 1.35,
                "detector_frame_mass_2": 1.25,
                "inclination": 0.3,
                "coa_time": 2001.0,
            },
        ]

    def test_from_mapping_preserves_mapping_order_without_renaming(self):
        """Direct population mappings keep their canonical key order untouched."""
        population_mapping = {
            "detector_frame_mass_1": np.array([35.0, 36.0]),
            "detector_frame_mass_2": np.array([25.0, 26.0]),
            "polarization_angle": np.array([0.4, 0.5]),
        }

        adapter = PopulationAdapter.from_mapping(population_mapping, source_type="bbh")

        assert adapter.parameter_names == tuple(population_mapping.keys())
        assert list(adapter.get_event_parameters(1).keys()) == list(population_mapping.keys())
        assert adapter.get_event_parameters(1) == {
            "detector_frame_mass_1": 36.0,
            "detector_frame_mass_2": 26.0,
            "polarization_angle": 0.5,
        }

    def test_from_mapping_rejects_mismatched_lengths(self):
        """All parameter arrays must describe the same number of events."""
        with pytest.raises(ValueError, match="same number of samples"):
            PopulationAdapter.from_mapping(
                {
                    "detector_frame_mass_1": np.array([30.0, 32.0]),
                    "detector_frame_mass_2": np.array([20.0]),
                },
                source_type="bbh",
            )

    def test_from_mapping_rejects_key_order_mismatches(self):
        """Explicit parameter order must match the mapping keys exactly."""
        with pytest.raises(ValueError, match="match parameter_names in the same order"):
            PopulationAdapter.from_mapping(
                {
                    "detector_frame_mass_1": np.array([30.0]),
                    "detector_frame_mass_2": np.array([20.0]),
                },
                source_type="bbh",
                parameter_names=("detector_frame_mass_2", "detector_frame_mass_1"),
            )

    @pytest.mark.parametrize(
        ("backend_name", "expected_class_name"),
        [
            ("bbh", "BBHSimulator"),
            ("cbc_prior", "CBCSimulator"),
            ("bns_prior", "BNSSimulator"),
            ("nsbh_prior", "NSBHSimulator"),
            ("file", "FilePopulationLoader"),
        ],
    )
    def test_instantiate_population_backend_resolves_builtin_aliases(
        self,
        backend_name: str,
        expected_class_name: str,
    ):
        """Built-in aliases should resolve through the public population resolver."""
        backend_class = resolve_backend_class("population", backend_name)

        assert backend_class.__name__ == expected_class_name

    def test_instantiate_population_backend_resolves_module_class(self):
        """Module:Class backends should resolve through the public population resolver."""
        backend_class = resolve_backend_class("population", "tests.population.test_adapter:ModulePopulationBackend")

        assert backend_class.__name__ == "ModulePopulationBackend"

    def test_instantiate_population_backend_resolves_entry_point(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Third-party entry points should resolve through the public population resolver."""
        alias = "population_test_adapter_alias"
        _install_entry_point_backend(
            tmp_path,
            monkeypatch,
            alias=alias,
            package_name="population_test_adapter_backend_pkg",
        )

        backend_class = resolve_backend_class("population", alias)
        backend = backend_class()

        assert isinstance(backend, GWPopSimulator)
        assert backend.__class__.__name__ == "EntryPointPopulationBackend"
        assert backend.simulate(1)["detector_frame_mass_1"][0] == pytest.approx(7.0)

    def test_population_mapping_property_returns_proxy(self):
        adapter = PopulationAdapter.from_mapping({"coa_time": np.array([1.0, 2.0])}, source_type="bbh")
        mapping = adapter.population_mapping
        assert mapping["coa_time"] == (1.0, 2.0)
        with pytest.raises((TypeError, AttributeError)):
            mapping["coa_time"] = np.array([9.0])  # type: ignore[index]

    def test_get_event_parameters_out_of_bounds(self):
        adapter = PopulationAdapter.from_mapping({"coa_time": np.array([1.0])}, source_type="bbh")
        with pytest.raises(IndexError, match="out of range"):
            adapter.get_event_parameters(5)

    def test_from_mapping_rejects_invalid_source_type(self):
        with pytest.raises(ValueError, match="non-empty string"):
            PopulationAdapter.from_mapping({"coa_time": np.array([1.0])}, source_type="")


class _ConversionSpy:
    """Stands in for a device array and records how it was read.

    The point is to distinguish *one bulk conversion* from *many element reads*, which is the whole
    claim. Inferring that from the stored value type does not work: a per-element loop that calls
    ``float()`` on each sample also leaves Python floats behind, so it passes a type check while
    costing what the bulk transfer exists to avoid.
    """

    def __init__(self, values: np.ndarray) -> None:
        self._values = values
        self.bulk_conversions = 0
        self.element_reads = 0

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        self.bulk_conversions += 1
        return np.asarray(self._values, dtype=dtype)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: Any) -> Any:
        self.element_reads += 1
        return self._values[index]

    def __iter__(self):
        for index in range(len(self._values)):
            self.element_reads += 1
            yield self._values[index]


class TestBulkHostTransfer:
    """``gwmock-pop`` returns JAX device arrays, and how they are read dominates this adapter."""

    def test_the_values_are_read_in_one_bulk_conversion(self):
        """The faithful witness: count the conversions, do not infer them from the stored type.

        A reviewer defeated the type-based check by writing a per-element implementation that calls
        ``float()`` on each sample -- Python floats in the store, every test passing, and 1.4 s per
        500 elements against 0.4 ms. Counting is what actually pins the behaviour.
        """
        spy = _ConversionSpy(np.arange(5.0))

        adapter = PopulationAdapter.from_mapping({"coa_time": spy}, source_type="bbh")

        assert spy.bulk_conversions == 1, f"converted {spy.bulk_conversions} times, expected exactly one"
        assert spy.element_reads == 0, (
            f"read {spy.element_reads} elements individually; on a device array each of those is its "
            f"own transfer, which is the cost this exists to avoid"
        )
        assert adapter.population_mapping["coa_time"] == (0.0, 1.0, 2.0, 3.0, 4.0)

    def test_device_arrays_are_transferred_in_bulk_not_element_by_element(self):
        """Pinned by the *type* of the stored values, because timing tests are flaky.

        A bare ``tuple(device_array)`` iterates the array, pulling each element back with its own
        device operation, and leaves JAX scalars behind. Converting in bulk first leaves plain Python
        floats. So the element type is a faithful witness for which path ran, and it fails
        immediately if the bulk conversion is removed.

        The cost this guards is not marginal: 0.994 ms per event against 0.0021 ms over a
        1000-event, eight-parameter catalogue, a factor of 469, and it was the entire construction
        cost of the adapter.
        """
        jax = pytest.importorskip("jax", reason="the [jax] extra is not installed")

        adapter = PopulationAdapter.from_mapping(
            {"coa_time": jax.numpy.asarray([1.0, 2.0, 3.0])},
            source_type="bbh",
        )

        stored = adapter.population_mapping["coa_time"]
        assert [type(value) for value in stored] == [float, float, float], (
            f"stored {[type(v).__name__ for v in stored]}; JAX scalars here mean the device array was "
            f"iterated element by element rather than transferred once"
        )
        assert stored == (1.0, 2.0, 3.0)

    def test_the_values_survive_the_bulk_conversion_unchanged(self):
        """Speed is worthless if the numbers move. Full float64 precision, not just close."""
        jax = pytest.importorskip("jax", reason="the [jax] extra is not installed")
        jax.config.update("jax_enable_x64", True)

        exact = [1577491296.123456789, -0.30000000000000004, 1e-21, 2.5e30]
        adapter = PopulationAdapter.from_mapping(
            {"value": jax.numpy.asarray(exact, dtype=jax.numpy.float64)},
            source_type="bbh",
        )

        stored = adapter.population_mapping["value"]
        assert stored == tuple(exact), f"bulk transfer changed the values: {stored} against {tuple(exact)}"

    def test_a_non_numeric_column_still_works(self):
        """Object columns cannot go through the numeric path, and must not crash it.

        This one checks behaviour, not the bulk path: it passes whichever conversion runs. Stated so
        the module is not read as three guards on the transfer when only the type witness above is
        one.
        """
        adapter = PopulationAdapter.from_mapping(
            {"label": ["a", "b"], "value": np.array([1.0, 2.0])},
            source_type="bbh",
        )

        assert adapter.get_event_parameters(1) == {"label": "b", "value": 2.0}

    def test_a_two_dimensional_column_is_refused_rather_than_reinterpreted(self):
        """A 2-D column used to become one entry per row, silently shortening the catalogue.

        ``tuple(values)`` on a 2-D array yields row arrays, and the shape check in
        ``_validate_parameter_values`` looks for a ``.shape`` attribute -- which a tuple does not
        have. So a (3, 2) column became a 3-event catalogue of array-valued parameters and no
        validation noticed. Pre-existing rather than introduced by the bulk transfer, but the bulk
        transfer is where the fix belongs: the array is passed through so the validator can see its
        shape.
        """
        with pytest.raises(ValueError, match="one-dimensional"):
            PopulationAdapter.from_mapping(
                {"coa_time": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])},
                source_type="bbh",
            )
