# Examples

This page provides an overview of example configuration files available for ET simulations.

## Overview

All example configurations in the [`examples/`](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/tree/main/examples) directory generate one day of data per detector, divided into 4096-seconds frames (21 frame files in total), sampled at 4096 Hz, starting from 1 January 2030.

For guidance on changing dataset duration or simulation properties, see the [Generating Data](generating-data.md) page.
For a more complete guide to writing your own configuration files, see the [Configuration Files](configuration.md) page.

To list all the available example configuration files:

```bash
gwsim config --list
```

To run any of the following configuration file:

```bash
# Copy configuration file to working directory
gwsim config --get <label> --output config.yaml

# Run simulation
gwsim simulate config.yaml
```

## Noise Generation

Example configurations for generating detector noise with various configurations and sensitivities.

**Einstein Telescope - Triangular**

- EMR location: [`noise/uncorrelated_gaussian/et_triangle_emr/config.yaml`](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/blob/main/examples/noise/uncorrelated_gaussian/et_triangle_emr/config.yaml)
- Sardinia location: [`noise/uncorrelated_gaussian/et_triangle_sardinia/config.yaml`](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/blob/main/examples/noise/uncorrelated_gaussian/et_triangle_sardinia/config.yaml)

**Einstein Telescope - 2L**

- Aligned configuration: [`noise/uncorrelated_gaussian/et_2l_aligned/config.yaml`](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/blob/main/examples/noise/uncorrelated_gaussian/et_2l_aligned/config.yaml)
- Misaligned configuration: [`noise/uncorrelated_gaussian/et_2l_misaligned/config.yaml`](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/blob/main/examples/noise/uncorrelated_gaussian/et_2l_misaligned/config.yaml)

## CBC Signals Generation

Example configurations for generating detector data with CBC signals with various configurations and sensitivities.

**Einstein Telescope - Triangular**

- EMR location:
  - BBH signals: [`signal/bbh/et_triangle_emr/config.yaml`](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/blob/main/examples/signal/bbh/et_triangle_emr/config.yaml)
  - BNS signals: [`signal/bns/et_triangle_emr/config.yaml`](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/blob/main/examples/signal/bns/et_triangle_emr/config.yaml)
- Sardinia location:
  - BBH signals: [`signal/bbh/et_triangle_sardinia/config.yaml`](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/blob/main/examples/signal/bbh/et_triangle_sardinia/config.yaml)
  - BNS signals: [`signal/bns/et_triangle_sardinia/config.yaml`](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/blob/main/examples/signal/bns/et_triangle_sardinia/config.yaml)

**Einstein Telescope - 2L**

- Aligned configuration:
  - BBH signals: [`signal/bbh/et_2l_aligned/config.yaml`](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/blob/main/examples/signal/bbh/et_2l_aligned/config.yaml)
  - BNS signals: [`signal/bns/et_2l_aligned/config.yaml`](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/blob/main/examples/signal/bns/et_2l_aligned/config.yaml)
- Misaligned configuration:
  - BBH signals: [`signal/bbh/et_2l_misaligned/config.yaml`](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/blob/main/examples/signal/bbh/et_2l_misaligned/config.yaml)
  - BNS signals: [`signal/bns/et_2l_misaligned/config.yaml`](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/blob/main/examples/signal/bns/et_2l_misaligned/config.yaml)

## Glitch Generation

Example configurations for generating detector glitches with various configurations and sensitivities.

**Einstein Telescope - Triangular**

- EMR location: [`glitch/gengli/et_triangle_emr/config.yaml`](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/blob/main/glitch/gengli/et_triangle_emr/config.yaml)
- Sardinia location: [`glitch/gengli/et_triangle_sardinia/config.yaml`](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/blob/main/examples/glitch/gengli/et_triangle_sardinia/config.yaml)

**Einstein Telescope - 2L**

- Aligned configuration: [`glitch/gengli/et_2l_aligned/config.yaml`](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/blob/main/examples/glitch/gengli/et_2l_aligned/config.yaml)
- Misaligned configuration: [`glitch/gengli/et_2l_misaligned/config.yaml`](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/blob/main/examples/examples/glitch/gengli/et_2l_misaligned/config.yaml)

## Storage Estimates

For reference, typical storage requirements:

- **Noise**: ~123 MB per file, ~7.6 GB for 3 detectors, 24 hours
- **Signals**: Variable depending on waveform complexity, typically similar to noise
- **Glitches**: Variable depending on number and duration
