# Examples

This page provides an overview of example configuration files available for ET
simulations.

## Overview

All example configurations in the
[`examples/`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples)
directory generate one day of data per detector, divided into 4096-seconds
frames (21 frame files in total), sampled at 4096 Hz, starting from 1
January 2030.

For guidance on changing dataset duration or simulation properties, see the
[Generating Data](generating-data.md) page. For a more complete guide to writing
your own configuration files, see the [Configuration Files](configuration.md)
page.

To list all the available example configuration files:

```bash
gwmock config --list
```

To run any of the following configuration files:

```bash
# Copy configuration file to working directory
gwmock config --get <label> --output config.yaml

# Run simulation
gwmock simulate config.yaml
```

## Blank Starting Config

The `default_config` label produces a minimal template that shows the full
`orchestration:` schema with a local `population.h5` placeholder. Use it as a
starting point when writing a custom configuration from scratch:

```bash
gwmock config --get default_config --output config.yaml
```

The generated file expects you to supply a local `population.h5` file. Replace
`path: population.h5` with an actual path or URL before running.

## Noise Generation

Example configurations for generating detector noise with various configurations
and sensitivities.

### Einstein Telescope - Triangular

- EMR location:
  [`noise/uncorrelated_gaussian/et_triangle_emr/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/uncorrelated_gaussian/et_triangle_emr/config.yaml)
- Sardinia location:
  [`noise/uncorrelated_gaussian/et_triangle_sardinia/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/uncorrelated_gaussian/et_triangle_sardinia/config.yaml)

### Einstein Telescope - 2L

- Aligned configuration:
  [`noise/uncorrelated_gaussian/et_2l_aligned/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/uncorrelated_gaussian/et_2l_aligned/config.yaml)
- Misaligned configuration:
  [`noise/uncorrelated_gaussian/et_2l_misaligned/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/uncorrelated_gaussian/et_2l_misaligned/config.yaml)

## CBC Signals Generation

Example configurations for generating detector data with BBH signals with
various configurations and sensitivities.

### Einstein Telescope - Triangular

- EMR location:
  [`signal/bbh/et_triangle_emr/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/signal/bbh/et_triangle_emr/config.yaml)
- Sardinia location:
  [`signal/bbh/et_triangle_sardinia/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/signal/bbh/et_triangle_sardinia/config.yaml)

### Einstein Telescope - 2L

- Aligned configuration:
  [`signal/bbh/et_2l_aligned/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/signal/bbh/et_2l_aligned/config.yaml)
- Misaligned configuration:
  [`signal/bbh/et_2l_misaligned/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/signal/bbh/et_2l_misaligned/config.yaml)

## Glitch Generation

Example configurations for generating detector glitches with various
configurations and sensitivities.

### Einstein Telescope - Triangular

- EMR location
    - E1:
      [`noise/glitches/gengli/et_triangle_emr/e1/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/gengli/et_triangle_emr/e1/config.yaml)
    - E2:
      [`noise/glitches/gengli/et_triangle_emr/e2/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/gengli/et_triangle_emr/e2/config.yaml)
    - E3:
      [`noise/glitches/gengli/et_triangle_emr/e3/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/gengli/et_triangle_emr/e3/config.yaml)
- Sardinia location
    - E1:
      [`noise/glitches/gengli/et_triangle_sardinia/e1/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/gengli/et_triangle_sardinia/e1/config.yaml)
    - E2:
      [`noise/glitches/gengli/et_triangle_sardinia/e2/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/gengli/et_triangle_sardinia/e2/config.yaml)
    - E3:
      [`noise/glitches/gengli/et_triangle_sardinia/e3/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/gengli/et_triangle_sardinia/e3/config.yaml)

### Einstein Telescope - 2L

- Aligned configuration
    - E1:
      [`noise/glitches/gengli/et_2l_aligned/e1/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/gengli/et_2l_aligned/e1/config.yaml)
    - E2:
      [`noise/glitches/gengli/et_2l_aligned/e2/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/gengli/et_2l_aligned/e2/config.yaml)
- Misaligned configuration
    - E1:
      [`noise/glitches/gengli/et_2l_misaligned/e1/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/gengli/et_2l_misaligned/e1/config.yaml)
    - E2:
      [`noise/glitches/gengli/et_2l_misaligned/e2/config.yaml`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples/noise/glitches/gengli/et_2l_misaligned/e2/config.yaml)

## Storage Estimates

For reference, typical storage requirements:

- **Noise**: ~123 MB per file, ~7.6 GB for 3 detectors, 24 hours
- **Signals**: Variable depending on waveform complexity, typically similar to
  noise
- **Glitches**: Variable depending on number and duration
