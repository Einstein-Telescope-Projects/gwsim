"""Unit tests for CBCSignalSimulator."""

from __future__ import annotations

from unittest.mock import patch

from gwsim.signal.cbc import CBCSignalSimulator


class TestCBCSignalSimulator:
    """Test CBCSignalSimulator initialization and inheritance."""

    def test_init_success(self, tmp_path):
        """Test successful initialization with valid CBC file."""
        dummy_file = tmp_path / "cbc_test.h5"
        dummy_file.write_bytes(b"dummy")

        with patch("h5py.File") as mock_file:
            mock_f = mock_file.return_value.__enter__.return_value
            mock_f.items.return_value = [
                ("tc", type("MockDataset", (), {"__getitem__": lambda self, idx: [100.0, 50.0]})()),
                ("m1", type("MockDataset", (), {"__getitem__": lambda self, idx: [30.0, 25.0]})()),
                ("m2", type("MockDataset", (), {"__getitem__": lambda self, idx: [25.0, 20.0]})()),
            ]
            mock_f.attrs.items.return_value = []

            simulator = CBCSignalSimulator(
                population_file=str(dummy_file),
                waveform_model="IMRPhenomD",
                start_time=0,
                duration=100.0,
                sampling_frequency=4096,
                detectors=["H1"],
                minimum_frequency=5.0,
            )
            assert simulator is not None
            assert simulator.population_data is not None
            # Verify CBC-specific sorting by tc
            assert list(simulator.population_data["tc"]) == [50.0, 100.0]

    def test_inheritance_uses_cbc_mixin(self, tmp_path):
        """Test that CBCSignalSimulator uses CBCPopulationReaderMixin for parameter mapping."""
        dummy_file = tmp_path / "cbc_mapping.h5"
        dummy_file.write_bytes(b"dummy")

        with patch("h5py.File") as mock_file:
            mock_f = mock_file.return_value.__enter__.return_value
            mock_f.items.return_value = [
                ("tc", type("MockDataset", (), {"__getitem__": lambda self, idx: [100.0]})()),
                ("m1", type("MockDataset", (), {"__getitem__": lambda self, idx: [30.0]})()),
                ("m2", type("MockDataset", (), {"__getitem__": lambda self, idx: [25.0]})()),
                ("z", type("MockDataset", (), {"__getitem__": lambda self, idx: [0.1]})()),
            ]
            mock_f.attrs.items.return_value = []

            simulator = CBCSignalSimulator(
                population_file=str(dummy_file),
                waveform_model="IMRPhenomD",
                start_time=0,
                duration=100.0,
                sampling_frequency=4096,
                detectors=["H1"],
                minimum_frequency=5.0,
            )
            # Check that CBC parameter mapping is applied (m1 -> mass1, z -> redshift)
            assert "mass1" in simulator.population_data.columns
            assert "redshift" in simulator.population_data.columns
            assert "m1" not in simulator.population_data.columns
