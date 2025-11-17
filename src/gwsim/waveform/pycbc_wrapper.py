"""Wrapper for generating waveforms using PyCBC and converting them to GWpy TimeSeries."""

from __future__ import annotations

from gwpy.timeseries import TimeSeries
from pycbc.waveform import get_td_waveform


def pycbc_waveform_wrapper(
    tc: float, sampling_frequency: float, minimum_frequency: float, waveform_model: str, **kwargs
) -> dict[str, TimeSeries]:
    """Wrapper to generate waveforms using PyCBC and convert to GWpy TimeSeries.

    Args:
        tc: Coalescence time in GPS seconds.
        sampling_frequency: Sampling frequency in Hz.
        minimum_frequency: Minimum frequency of the waveform in Hz.
        waveform_model: Name of the waveform model to use.
        **kwargs: Additional keyword arguments for the waveform generation.
    Returns:
        A dictionary with 'plus' and 'cross' keys containing the respective GWpy TimeSeries.
    """

    # Call PyCBC to generate the waveform.
    hp, hc = get_td_waveform(
        approximant=waveform_model, delta_t=1 / sampling_frequency, f_lower=minimum_frequency, **kwargs
    )

    # Add the coalescence time.
    hp.start_time += tc
    hc.start_time += tc

    # Convert to GWpy
    hp_gwpy = TimeSeries.from_pycbc(hp, copy=True)
    hc_gwpy = TimeSeries.from_pycbc(hc, copy=True)

    return {"plus": hp_gwpy, "cross": hc_gwpy}
