"""Integration tests for pycbc_waveform_wrapper with real PyCBC calls."""

from __future__ import annotations

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform

from gwsim.waveform.pycbc_wrapper import pycbc_waveform_wrapper

# Mark as integration tests - slower, requires PyCBC
pytestmark = pytest.mark.integration


class TestPyCBCWaveformWrapperIntegration:
    """Integration tests with real PyCBC waveform generation."""

    @pytest.mark.slow
    def test_real_waveform_generation(self):
        """Test actual waveform generation with IMRPhenomD."""
        sampling_frequency = 4096

        result = pycbc_waveform_wrapper(
            tc=1234567890.0,
            sampling_frequency=sampling_frequency,
            waveform_model="IMRPhenomD",
            mass1=40.0,
            mass2=30.0,
            spin1z=0.5,
            spin2z=-0.3,
            minimum_frequency=20.0,
        )

        pycbc_hp, _pycbc_hc = get_td_waveform(
            mass1=40.0,
            mass2=30.0,
            spin1z=0.5,
            spin2z=-0.3,
            f_lower=20.0,
            delta_t=1.0 / sampling_frequency,
            approximant="IMRPhenomD",
        )

        # Verify structure
        assert isinstance(result, dict)
        assert "plus" in result
        assert "cross" in result

        # Verify types
        assert isinstance(result["plus"], TimeSeries)
        assert isinstance(result["cross"], TimeSeries)

        # Verify waveform properties
        assert result["plus"].sample_rate.value == sampling_frequency
        assert result["cross"].sample_rate.value == sampling_frequency
        assert result["plus"].t0.value == float(pycbc_hp.start_time + 1234567890.0)
        assert result["cross"].t0.value == float(pycbc_hp.start_time + 1234567890.0)

        # Verify data is non-zero
        assert result["plus"].max().value > 0
        assert result["cross"].max().value > 0

    @pytest.mark.slow
    def test_pycbc_to_gwpy_conversion_integration(self):
        """Test that PyCBC to GWpy conversion preserves data properties."""
        result = pycbc_waveform_wrapper(
            tc=1234567890.0,
            sampling_frequency=4096,
            waveform_model="IMRPhenomD",
            mass1=40.0,
            mass2=30.0,
            minimum_frequency=20.0,
        )

        # Check that GWpy conversion preserved the data
        assert not np.isnan(result["plus"].data).any()
        assert not np.isnan(result["cross"].data).any()
        assert not np.isinf(result["plus"].data).any()
        assert not np.isinf(result["cross"].data).any()
