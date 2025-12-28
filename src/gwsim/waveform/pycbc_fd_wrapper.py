"""
PyCBC frequency-domain waveform generation utilities.

This module provides a thin wrapper around PyCBC's frequency-domain
waveform generator, returning GWpy FrequencySeries objects with
additional metadata needed for later time-domain conversion.
"""

from __future__ import annotations

from gwpy.frequencyseries import FrequencySeries
from pycbc.waveform import get_fd_waveform


def pycbc_fd_waveform_wrapper(
    tc: float,
    duration: float,
    sampling_frequency: float,
    minimum_frequency: float,
    waveform_model: str,
    **kwargs,
) -> dict[str, FrequencySeries]:
    """Generate frequency-domain waveforms using PyCBC.

    Args:
        tc: Coalescence time in GPS seconds (stored, not applied yet).
        duration: Total duration of the waveform in seconds.
        sampling_frequency: Sampling frequency in Hz.
        minimum_frequency: Minimum frequency of the waveform in Hz.
        waveform_model: Name of the waveform approximant.
        **kwargs: Additional waveform parameters.

    Returns:
        Dictionary with 'plus' and 'cross' FrequencySeries.
    """

    delta_f = 1.0 / duration

    hp, hc = get_fd_waveform(
        approximant=waveform_model,
        delta_f=delta_f,
        f_lower=minimum_frequency,
        **kwargs,
    )

    # Convert to GWpy FrequencySeries
    hp_fs = FrequencySeries.from_pycbc(hp, copy=True)
    hc_fs = FrequencySeries.from_pycbc(hc, copy=True)

    # Store metadata for later TD conversion
    hp_fs.meta["tc"] = tc
    hp_fs.meta["duration"] = duration
    hp_fs.meta["sampling_frequency"] = sampling_frequency

    hc_fs.meta.update(hp_fs.meta)

    return {"plus": hp_fs, "cross": hc_fs}
<<<<<<< HEAD

=======
>>>>>>> 20c54d9 (fix: pre-commit issues resolved)
