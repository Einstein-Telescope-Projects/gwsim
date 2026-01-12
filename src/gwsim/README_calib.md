## Frequency-Domain Detector Calibration

This section describes how frequency-dependent detector calibration errors are implemented in `gwsim`, how they are configured, and how they are applied in the simulation pipeline.

### Motivation

For long-duration signals at low frequencies, such as those relevant for ET-class detectors, waveform generation dominates the wall time. Many waveform models are implemented internally in the frequency domain, and generating time-domain waveforms already involves an internal inverse Fourier transform.

To avoid redundant FFT and IFFT operations, calibration errors are applied directly in the frequency domain before any time-domain conversion. This ensures that only a single inverse Fourier transform is required per waveform.

Calibration is treated as a detector property and is fully configured through the configuration file.

### Overview of Changes

The following functionality was added:

- A new frequency-domain calibration model
- Support for detector-specific calibration through the configuration system
- A detector-level hook to apply calibration in the frequency domain
- No changes to existing time-domain detector projection logic

Earth rotation, antenna pattern evaluation, and time delays are handled exactly as before.

### Calibration Model

A new file was added:

This file defines a `CalibrationModel` class that represents fixed, frequency-dependent calibration errors.

The model stores:

- A frequency grid
- Fractional amplitude errors
- Phase errors in radians

Amplitude and phase errors are interpolated linearly in frequency. Outside the provided frequency range, errors default to zero.

The calibration transfer function is defined in https://dcc.ligo.org/public/0116/T1400682/001/calnote.pdf

Detectors can now be specified either as strings or as structured entries with calibration information.

Example detector configuration with calibration:

```yaml
detectors:
  - name: E1_triangle_emr
    calibration:
      file: calibration/E1_calibration.txt

  - name: E2_triangle_emr
    calibration:
      file: calibration/E2_calibration.txt

  - name: E3_triangle_emr
    calibration:
      file: calibration/E3_calibration.txt


Calibration files are plain text files with three columns (freq, amp, phase) and this might be too basic of an assumption.

## Frequency-Domain Waveform Generation (`waveform.py`)

Waveform generation was updated to use PyCBC’s frequency-domain interface.

Instead of calling `get_td_waveform`, waveforms are generated using:

- `pycbc.waveform.get_fd_waveform`

This produces plus and cross polarizations directly in the frequency domain.

Key points:
- Waveforms are generated on a uniform frequency grid
- No inverse Fourier transform is performed at this stage
- Metadata required for later time-domain conversion is preserved
- This avoids the internal IFFT that would otherwise occur when generating time-domain waveforms

The frequency-domain waveforms are passed downstream for calibration before any conversion to the time domain.



## Detector-Level Changes (`detector.py`)

### Detector Configuration Parsing

The detector configuration logic was extended to support structured detector entries.

Detectors can now be specified either as simple strings or as dictionaries containing detector-specific options, including calibration.

This change is backward compatible with existing configuration files.

When a calibration block is present, the detector loads a calibration model at initialization time and stores it as a detector attribute.

```
