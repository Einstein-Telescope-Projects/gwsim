# Generating Data

This guide shows how to use `gwsim` to create realistic mock data for gravitational-wave detectors.

It uses the Einstein Telescope (ET) triangular configuration located in the Meuse-Rhine Euregion as an example.

For an overview of all example configuration files for ET simulations, see the [Examples](examples.md) page.
For a quick guide on reading and working with the output GWF files, see the [Reading Data](reading-data.md) page.

## Generating Detector Noise

Detector noise can be generated using configuration files in the [`examples/noise`](https://github.com/Leuven-Gravity-Institute/gwsim/tree/main/examples/noise) directory.
An example configuration for producing two hours of ET noise data is provided in [`uncorrelated_gaussian/et_triangle_emr/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwsim/blob/main/examples/noise/uncorrelated_gaussian/et_triangle_emr/config.yaml):

```yaml
--8<-- "examples/noise/uncorrelated_gaussian/et_triangle_emr/config.yaml"
```

This configuration generates one day of noise data per detector (E1, E2, E3).
Each frame file covers 4096 seconds, resulting in 22 frame files, starting on 1 January 2030.

Noise is simulated using the [ET_10_full_cryo_psd](https://github.com/Leuven-Gravity-Institute/gwsim/blob/main/src/gwsim/detector/noise_curves/ET_10_full_cryo_psd.txt) sensitivity curve from the [CoBA Science Study](https://iopscience.iop.org/article/10.1088/1475-7516/2023/07/068) and publicly available.
A low-frequency cutoff of 2 Hz is used.

To generate the ET noise data, run:

```bash
# Create working directory
mkdir noise_et_triangle_emr
cd noise_et_triangle_emr

# Copy configuration file to your working directory
gwsim config --get noise/uncorrelated_gaussian/et_triangle_emr --output config.yaml

# Run simulation
gwsim simulate config.yaml
```

#### Storage Requirements

Each GWF file is approximately 123 MB. For three detectors with 21 files each:

- **Data files**: ~7.6 GB
- **Metadata**: ~52.5 KB
- **Total**: ~7.6 GB

## Generating CBC Signals

Compact Binary Coalescence (CBC) signals can be generated using configuration files in the [`examples/signal/bbh`](https://github.com/Leuven-Gravity-Institute/gwsim/tree/main/examples/signal/bbh) and [`examples/signal/bns`](https://github.com/Leuven-Gravity-Institute/gwsim/tree/main/examples/signal/bns) directories.

### Binary Black Hole (BBH) Signals

An example configuration for producing one day of ET data containing BBH signals from a realistic population is provided in [`signal/bbh/et_triangle_emr/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwsim/tree/main/examples/signal/bbh/et_triangle_emr/config.yaml):

```yaml
--8<-- "examples/signal/bbh/et_triangle_emr/config.yaml"
```

As with the noise example, this configuration file produces one day of data per detectors, with each frame file lasting 4096 seconds (for a total of 21 frame files), starting on 1 January 2030.

BBH signals are injected into zero noise from the [18321_1yrCatalogBBH.h5](https://apps.et-gw.eu/tds/?content=3&r=18321) population file used in the CoBA study and publicly available.
The [IMRPhenomXPHM](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.104056) waveform model is used, with a low-frequency cutoff of 2 Hz and including Earth rotation effects.

To generate the ET data with BBH signals, run:

```bash
# Create working directory
mkdir bbh_et_triangle_emr
cd bbh_et_triangle_emr

# Copy configuration file to your working directory
gwsim config --get signal/bbh/et_triangle_emr --output config.yaml

# Run simulation
gwsim simulate config.yaml
```

<!-- prettier-ignore -->
!!! note
    The configuration file automatically downloads the BBH population file from a [Zenodo repository](https://sandbox.zenodo.org/records/413548).
    The file is saved in a cache directory (by default, `~/.gwsim/population/`).
    When the same population file is needed again, gwsim uses the cached copy to avoid re-downloading.

### Binary Neutron Star (BNS) Signals

An example configuration for producing one day of ET data containing BNS signals from a realistic population is provided in [`signal/bns/et_triangle_emr/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwsim/tree/main/examples/signal/bns/et_triangle_emr/config.yaml).
It is equivalent to the BBH example configuration, except for:

```yaml
population_file: https://sandbox.zenodo.org/records/413548/files/18321_1yrCatalogBNS.h5
waveform_model: IMRPhenomPv2_NRTidalv2
```

BNS signals are injected into zero noise from the [18321_1yrCatalogBNS.h5](https://apps.et-gw.eu/tds/?content=3&r=18321) population file used in the CoBA study and publicly available.
The [IMRPhenomPv2_NRTidalv2](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.100.044003) waveform model is used, with a low-frequency cutoff of 2 Hz and including Earth rotation effects.

To generate the ET data with BNS signals, run:

```bash
# Create working directory
mkdir bns_et_triangle_emr
cd bns_et_triangle_emr

# Copy configuration file to your working directory for BNS simulation
gwsim config --get signal/bns/et_triangle_emr --output config.yaml

# Run simulation
gwsim simulate config.yaml
```

<!-- prettier-ignore -->
!!! note
    The configuration file automatically downloads the BNS population file from a [Zenodo repository](https://sandbox.zenodo.org/records/413548).
    The file is saved in a cache directory (by default, `~/.gwsim/population/`).
    When the same population file is needed again, `gwsim` uses the cached copy to avoid re-downloading.

## Generating Transient Noise Artifacts (Glitches)

Glitches can be generated using configuration files in the [`examples/glitch`](https://github.com/Leuven-Gravity-Institute/gwsim/tree/main/examples/glitch/gengli) directory.
Glitch generation uses the [gengli](https://pypi.org/project/gengli/) package and currently supports only _blip_ glitches.

An example configuration for producing one day of ET data for the E1 detector containing blip glitches from a realistic population is provided in [`glitch/gengli/et_triangle_emr/e1/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwsim/tree/main/examples/glitch/gengli/et_triangle_emr/e1/config.yaml):

```yaml
--8<-- "examples/glitch/gengli/et_triangle_emr/e1/config.yaml"
```

This configuration file generates one day of data for the E1 detector, divided into 4096-second frame files (for a total of 21 frames), starting on 1 January 2030.

Blip glitches are injected into zero noise from the [blip_glitch_population_E1.h5](https://sandbox.zenodo.org/records/413548) population file, which was generated with gengli using the [following script](https://github.com/Leuven-Gravity-Institute/gwsim/blob/main/src/gwsim/population/glitch.py).
These glitches are modeled on LIGO blip glitches observed during the O3 observing run and recolored to match the ET sensitivity.

To generate the ET data for detector E1 with glitches, run:

```bash
# Create working directory
mkdir -p glitch_et_triangle_emr/e1
cd glitch_et_triangle_emr/e1

# Copy configuration file to your working directory for glitch simulation
gwsim config --get glitch/gengli/et_triangle_emr/e1 --output config.yaml

# Run simulation
gwsim simulate config.yaml
```

<!-- prettier-ignore -->
!!! note
    The configuration file automatically downloads the glitch population file from a [Zenodo repository](https://sandbox.zenodo.org/records/413548).
    The file is saved in a cache directory (by default, `~/.gwsim/population/`).
    When the same population file is needed again, gwsim uses the cached copy to avoid re-downloading.

<!-- prettier-ignore -->
!!! note
    The [`GengliGlitchSimulator`](/reference/gwsim/glitch/gengli_glitch) currently supports only a single detector at a time.
    To generate glitch-containing data for detectors E2 and E3, rerun the command above using the [`glitch/gengli/et_triangle_emr/e2/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwsim/tree/main/examples/glitch/gengli/et_triangle_emr/e2/config.yaml) and [`glitch/gengli/et_triangle_emr/e3/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwsim/tree/main/examples/glitch/gengli/et_triangle_emr/e3/config.yaml) configuration files respectively (updating the working directory name).
    Note that a different glitch population is used for each detector.

## Using Different Detector Configurations

`gwsim` includes several pre-configured Einstein Telescope detector geometries, available in [`gwsim/detector/detectors`](https://github.com/Leuven-Gravity-Institute/gwsim/tree/main/src/gwsim/detector/detectors):

Triangular Configuration (Meuse-Rhine Euregion)

- `E1_triangle_emr`
- `E2_triangle_emr`
- `E3_triangle_emr`

Triangular Configuration (Sardinia)

- `E1_triangle_sardinia`
- `E2_triangle_sardinia`
- `E3_triangle_sardinia`

2L Aligned Configuration

- `E1_2L_aligned_sardinia`
- `E2_2L_aligned_emr`

2L Misaligned Configuration

- `E1_2L_misaligned_sardinia`
- `E2_2L_misaligned_emr`

To use a specific configuration, update the `detectors` list in your configuration file:

```yaml
detectors:
  - E1_2L_aligned_sardinia
  - E2_2L_aligned_emr
```

You don't need to include all detectors. For example, to generate only E1 data:

```yaml
detectors:
  - E1_2L_aligned_sardinia
```

## Using Different Sensitivity Curves

Multiple Einstein Telescope sensitivity curves (PSD files) are available in [`gwsim/detector/noise_curves/`](https://github.com/Leuven-Gravity-Institute/gwsim/tree/main/src/gwsim/detector/noise_curves).
These correspond to those used in the CoBA study.

To use a specific sensitivity curve:

```yaml
simulators:
  noise:
    arguments:
      psd: ET_15_HF_psd.txt
```

<!-- prettier-ignore -->
!!! note
    The detector geometries assume 10 km arms for triangular configurations and 15 km arms for 2L configurations. Choose sensitivity curves accordingly.

## Adjusting Dataset Duration

The length of a dataset is controlled by:

```yaml
start-time: # GPS start time of the dataset
duration: # Duration per frame file (seconds)
total-duration: # Total duration of the dataset
```

To change the dataset duration, simply adjust these parameters in your configuration file.

You can also change the sampling frequency of your dataset (the number of samples per second, measured in Hz), using the `sampling-frequency` argument.

**Total number of frame files:**

The total number of frame files depends on the duration of each frame file and the total duration of the dataset, and it's rounded up to the next integer:

```
max_samples = ceil(total-duration / duration)
```

<!-- prettier-ignore-start -->

!!! note
    The `total-duration` argument can be passed as a `float` in seconds, or as a `str` specifying the time unit (`"1 day"`, `"5 days"`, `"2 weeks"`, `"2 months"`, etc.).
    The supported time units are:

    - `second`
    - `minute`
    - `hour`
    - `day`
    - `week`
    - `month` (30 days)
    - `year` (365 days).

    Singular and plural forms are both accepted (e.g., `"1 day"` and `"2 days"`).

<!-- prettier-ignore-end -->

<!-- prettier-ignore -->
!!! tip
    A [UTC/GPS time converter](https://gwosc.org/gps/) is available at the Gravitational Wave Open Science Center.

<!-- prettier-ignore-start -->

!!! tip
    Sampling frequencies are often powers of 2 for efficiency. Common choices:

    - 4096 Hz (standard for GW data analysis)
    - 2048 Hz
    - 16384 Hz (high-frequency instruments)

    Lowering sampling frequency reduces computation time but also reduces the highest resolvable frequency (Nyquist limit = sampling_frequency / 2).

<!-- prettier-ignore-end -->

## Generate Multi-Detector Correlated Noise

You can generate multi-detector correlated noise by specifying a cross-power spectral density (CSD) file:

<!-- prettier-ignore -->
!!! warning
    The example configuration file is not fully tested yet. Use at your own risk.

```yaml
globals:
  simulator-arguments:
    sampling-frequency: 4096
    duration: 4096
    total-duration: '1 day'
    start-time: 1577491218
  working-directory: './ET_Triangle_EMR_correlated_noise'
  output-directory: 'data'
  metadata-directory: 'metadata'

simulators:
  noise:
    class: CorrelatedNoiseSimulator
    arguments:
      psd_file: ET_10_full_cryo_psd.txt
      csd_file: path_to_csd_file.txt
      detectors:
        - E1_Triangle_EMR
        - E2_Triangle_EMR
        - E3_Triangle_EMR
      low_frequency_cutoff: 2
      seed: 42
    output:
      file_name: 'E-{{ detectors }}_CORRELATED-NOISE_STRAIN-{{ start_time }}-{{ duration }}.gwf'
      arguments:
        channel: '{{ detectors }}:STRAIN'
```

`gwsim` uses a windowing approach to generate long-duration datasets.
If the input CSD varies rapidly with frequency, this windowing can introduce artifacts in the resulting frame files.

A diagnostic tool to check whether your CSD file is susceptible to such issues will be provided soon.

## Resume Interrupted Simulations

If a simulation is interrupted, resume it by running the same command:

```bash
# Start
gwsim simulate config.yaml

# If interrupted, resume
gwsim simulate config.yaml
```

gwsim automatically detects and continues from the last checkpoint.

## Combining Data Types

To create realistic mock data, you may generate noise, signals, and glitches separately, then combine them:

```bash
gwsim simulate noise_config.yaml
gwsim simulate signal_config.yaml
gwsim simulate glitch_config.yaml
```

Then merge the files using GWpy (see [Reading Data](reading-data.md) for details).
