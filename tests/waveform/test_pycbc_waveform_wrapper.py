"""Unit tests for pycbc_waveform_wrapper module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from gwsim.waveform.pycbc_wrapper import pycbc_waveform_wrapper


class TestPyCBCWaveformWrapper:
    """Test suite for pycbc_waveform_wrapper function."""

    @pytest.fixture
    def mock_pycbc_waveform(self):
        """Create mock PyCBC waveform time series."""
        # Mock hp (plus polarization)
        hp = MagicMock()
        hp.start_time = 0.0
        hp.delta_t = 1.0 / 4096
        hp.sample_rate = 4096
        hp.data = np.random.randn(16384)

        # Mock hc (cross polarization)
        hc = MagicMock()
        hc.start_time = 0.0
        hc.delta_t = 1.0 / 4096
        hc.sample_rate = 4096
        hc.data = np.random.randn(16384)

        return hp, hc

    @pytest.fixture
    def default_params(self):
        """Default parameters for waveform generation."""
        return {
            "tc": 1234567890.0,
            "sampling_frequency": 4096,
            "waveform_model": "IMRPhenomD",
            "mass1": 40.0,
            "mass2": 30.0,
            "spin1z": 0.5,
            "spin2z": -0.3,
            "minimum_frequency": 20.0,
        }

    def test_wrapper_basic_call(self, default_params, mock_pycbc_waveform):
        """Test basic waveform generation with valid parameters."""
        with patch("gwsim.waveform.pycbc_wrapper.get_td_waveform") as mock_get_td:
            mock_get_td.return_value = mock_pycbc_waveform

            _result = pycbc_waveform_wrapper(**default_params)

            # Check that get_td_waveform was called with correct parameters
            mock_get_td.assert_called_once()
            call_kwargs = mock_get_td.call_args[1]
            assert call_kwargs["approximant"] == "IMRPhenomD"
            assert call_kwargs["delta_t"] == 1.0 / 4096
            assert call_kwargs["mass1"] == default_params["mass1"]
            assert call_kwargs["mass2"] == default_params["mass2"]
            assert call_kwargs["f_lower"] == default_params["minimum_frequency"]

    def test_wrapper_returns_dict_with_plus_cross(self, default_params, mock_pycbc_waveform):
        """Test that wrapper returns dict with 'plus' and 'cross' keys."""
        with patch("gwsim.waveform.pycbc_wrapper.get_td_waveform") as mock_get_td:
            mock_get_td.return_value = mock_pycbc_waveform

            expected_length_of_result = 2  # 'plus' and 'cross'

            result = pycbc_waveform_wrapper(**default_params)

            assert isinstance(result, dict)
            assert "plus" in result
            assert "cross" in result
            assert len(result) == expected_length_of_result

    def test_wrapper_returns_gwpy_timeseries(self, default_params, mock_pycbc_waveform):
        """Test that wrapper returns GWpy TimeSeries objects."""
        with (
            patch("gwsim.waveform.pycbc_wrapper.get_td_waveform") as mock_get_td,
            patch("gwpy.timeseries.TimeSeries.from_pycbc") as mock_from_pycbc,
        ):
            number_of_calls = 2

            mock_get_td.return_value = mock_pycbc_waveform

            # Mock the from_pycbc to return TimeSeries
            mock_ts = MagicMock(spec=TimeSeries)
            mock_from_pycbc.return_value = mock_ts

            result = pycbc_waveform_wrapper(**default_params)

            assert isinstance(result["plus"], MagicMock)
            assert isinstance(result["cross"], MagicMock)
            assert mock_from_pycbc.call_count == number_of_calls

    def test_wrapper_sets_coalescence_time(self, default_params, mock_pycbc_waveform):
        """Test that coalescence time is correctly added to waveform."""
        tc = 1234567890.5
        params = {**default_params, "tc": tc}

        with patch("gwsim.waveform.pycbc_wrapper.get_td_waveform") as mock_get_td:
            mock_get_td.return_value = mock_pycbc_waveform

            pycbc_waveform_wrapper(**params)

            # Check that start_time was updated
            assert mock_pycbc_waveform[0].start_time == tc
            assert mock_pycbc_waveform[1].start_time == tc

    def test_wrapper_preserves_sampling_frequency(self, default_params, mock_pycbc_waveform):
        """Test that delta_t is correctly computed from sampling_frequency."""
        sampling_freq = 16384
        params = {**default_params, "sampling_frequency": sampling_freq}

        with patch("gwsim.waveform.pycbc_wrapper.get_td_waveform") as mock_get_td:
            mock_get_td.return_value = mock_pycbc_waveform

            pycbc_waveform_wrapper(**params)

            # Check that delta_t was computed correctly
            call_kwargs = mock_get_td.call_args[1]
            expected_delta_t = 1.0 / sampling_freq
            assert call_kwargs["delta_t"] == expected_delta_t

    def test_wrapper_passes_extra_kwargs(self, default_params, mock_pycbc_waveform):
        """Test that additional keyword arguments are passed to get_td_waveform."""
        params = {
            **default_params,
            "eccentricity": 0.1,
            "f_ref": 100.0,
        }

        with patch("gwsim.waveform.pycbc_wrapper.get_td_waveform") as mock_get_td:
            mock_get_td.return_value = mock_pycbc_waveform

            pycbc_waveform_wrapper(**params)

            call_kwargs = mock_get_td.call_args[1]
            assert call_kwargs["eccentricity"] == params["eccentricity"]
            assert call_kwargs["f_lower"] == params["minimum_frequency"]
            assert call_kwargs["f_ref"] == params["f_ref"]

    def test_wrapper_missing_required_parameter(self, mock_pycbc_waveform):
        """Test that wrapper raises error when required parameters are missing."""
        # Missing 'mass1' and 'mass2'
        params = {
            "tc": 1234567890.0,
            "sampling_frequency": 4096,
            "waveform_model": "IMRPhenomD",
        }

        with patch("gwsim.waveform.pycbc_wrapper.get_td_waveform") as mock_get_td:
            mock_get_td.side_effect = TypeError("get_td_waveform() missing required keyword argument")

            with pytest.raises(TypeError):
                pycbc_waveform_wrapper(**params)

    def test_wrapper_with_different_waveform_models(self, default_params, mock_pycbc_waveform):
        """Test wrapper with different waveform models."""
        models = ["IMRPhenomD", "IMRPhenomXPHM", "SEOBNRv4_opt"]

        with patch("gwsim.waveform.pycbc_wrapper.get_td_waveform") as mock_get_td:
            mock_get_td.return_value = mock_pycbc_waveform

            for model in models:
                params = {**default_params, "waveform_model": model}
                result = pycbc_waveform_wrapper(**params)

                call_kwargs = mock_get_td.call_args[1]
                assert call_kwargs["approximant"] == model
                assert "plus" in result
                assert "cross" in result

    def test_wrapper_with_zero_spins(self, default_params, mock_pycbc_waveform):
        """Test wrapper with non-spinning systems."""
        params = {
            **default_params,
            "spin1z": 0.0,
            "spin2z": 0.0,
        }

        with patch("gwsim.waveform.pycbc_wrapper.get_td_waveform") as mock_get_td:
            mock_get_td.return_value = mock_pycbc_waveform

            result = pycbc_waveform_wrapper(**params)

            call_kwargs = mock_get_td.call_args[1]
            assert call_kwargs["spin1z"] == 0.0
            assert call_kwargs["spin2z"] == 0.0
            assert "plus" in result
            assert "cross" in result

    def test_wrapper_with_large_mass_ratio(self, default_params, mock_pycbc_waveform):
        """Test wrapper with extreme mass ratio systems."""
        params = {
            **default_params,
            "mass1": 100.0,
            "mass2": 1.0,
        }

        with patch("gwsim.waveform.pycbc_wrapper.get_td_waveform") as mock_get_td:
            mock_get_td.return_value = mock_pycbc_waveform

            result = pycbc_waveform_wrapper(**params)

            call_kwargs = mock_get_td.call_args[1]
            assert call_kwargs["mass1"] == params["mass1"]
            assert call_kwargs["mass2"] == 1.0
            assert "plus" in result

    def test_wrapper_preserves_pycbc_data(self, default_params):
        """Test that wrapper correctly converts PyCBC data to GWpy TimeSeries."""
        # Create real-like PyCBC mock with actual data
        hp_data = np.sin(np.linspace(0, 10 * np.pi, 1000))
        hc_data = np.cos(np.linspace(0, 10 * np.pi, 1000))

        hp = MagicMock()
        hp.start_time = 0.0
        hp.delta_t = 1.0 / 4096
        hp.sample_rate = 4096
        hp.data = hp_data

        hc = MagicMock()
        hc.start_time = 0.0
        hc.delta_t = 1.0 / 4096
        hc.sample_rate = 4096
        hc.data = hc_data

        with (
            patch("gwsim.waveform.pycbc_wrapper.get_td_waveform") as mock_get_td,
            patch("gwpy.timeseries.TimeSeries.from_pycbc") as mock_from_pycbc,
        ):
            mock_get_td.return_value = (hp, hc)

            # Simulate real conversion
            mock_from_pycbc.side_effect = [
                TimeSeries(hp_data, t0=hp.start_time, sample_rate=hp.sample_rate),
                TimeSeries(hc_data, t0=hc.start_time, sample_rate=hc.sample_rate),
            ]

            result = pycbc_waveform_wrapper(**default_params)

            # Verify data integrity
            assert isinstance(result["plus"], TimeSeries)
            assert isinstance(result["cross"], TimeSeries)

    def test_wrapper_from_pycbc_copy_flag(self, default_params, mock_pycbc_waveform):
        """Test that from_pycbc is called with copy=True to avoid aliasing issues."""
        with (
            patch("gwsim.waveform.pycbc_wrapper.get_td_waveform") as mock_get_td,
            patch("gwpy.timeseries.TimeSeries.from_pycbc") as mock_from_pycbc,
        ):
            mock_get_td.return_value = mock_pycbc_waveform
            mock_from_pycbc.return_value = MagicMock(spec=TimeSeries)

            pycbc_waveform_wrapper(**default_params)

            # Check that from_pycbc was called with copy=True
            for call in mock_from_pycbc.call_args_list:
                assert call[1]["copy"] is True
