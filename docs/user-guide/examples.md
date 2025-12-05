# Examples

This page provides an overview of example configuration files available for ET simulations.

## Overview

All example configurations in the [`examples/`]() directory generate one day of data per detector, divided into 4096-seconds frames (22 frame files in total), sampled at 4096 Hz,  starting from 1 January 2030.

For guidance on changing dataset duration or simulation properties, see the [Generating Data](generating-data.md) page.
For a more complete guide to writing your own configuration files, see the [Configuration Files](configuration.md) page.

To run any of the following configuration file:

```bash
gwsim simulate config_file_name.yaml
```

## Noise Generation

Example configurations for generating detector noise with various configurations and sensitivities.

**Einstein Telescope - Triangular**

- EMR location: [`ET_Triangle_EMR_noise_config.yaml`]()
- Sardinia location: [`ET_Triangle_Sardinia_noise_config.yaml`]()

**Einstein Telescope - 2L**

- Aligned configuration: [`E1_2L_Aligned_noise_config.yaml`]()
- Misaligned configuration: [`E1_2L_Misaligned_noise_config.yaml`]()

## CBC Signals Generation

Example configurations for generating detector data with CBC signals with various configurations and sensitivities.

**Einstein Telescope - Triangular**

- EMR location:
    - BBH signals: [`ET_Triangle_EMR_BBH_config.yaml`]()
    - BNS signals: [`ET_Triangle_EMR_BNS_config.yaml`]()
- Sardinia location:
    - BBH signals: [`ET_Triangle_Sardinia_BBH_config.yaml`]()
    - BNS signals: [`ET_Triangle_Sardinia_BNS_config.yaml`]()

**Einstein Telescope - 2L**

- Aligned configuration:
    - BBH signals: [`E1_2L_Aligned_BBH_config.yaml`]()
    - BNS signals: [`E1_2L_Aligned_BNS_config.yaml`]()
- Misaligned configuration:
    - BBH signals: [`E1_2L_Misaligned_BBH_config.yaml`]()
    - BNS signals: [`E1_2L_Misaligned_BNS_config.yaml`]()

## Glitch Generation

Example configurations for generating detector glitches with various configurations and sensitivities.

**Einstein Telescope - Triangular**

- EMR location: [`ET_Triangle_EMR_glitch_config.yaml`]()
- Sardinia location: [`ET_Triangle_Sardinia_glitch_config.yaml`]()

**Einstein Telescope - 2L**

- Aligned configuration: [`E1_2L_Aligned_glitch_config.yaml`]()
- Misaligned configuration: [`E1_2L_Misaligned_glitch_config.yaml`]()

## Storage Estimates

For reference, typical storage requirements:

- **Noise**: ~125 MB per file, ~8.3 GB for 3 detectors, 24 hours
- **Signals**: Variable depending on waveform complexity, typically similar to noise
- **Glitches**: Variable depending on number and duration
