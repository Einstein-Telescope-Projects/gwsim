# Quick Start

This guide will help you run your first gwsim simulation in just a few minutes.

## 1. Run Your First Simulation

Run the following command in your working directory:

```bash
# Copy quick-start configuration file to working directory
gwsim config --get noise/uncorrelated_gaussian/quick_start --output quick_start_config.yaml

# Run simulation
gwsim simulate quick_start_config.yaml
```

The configuration file `quick_start_config.yaml` generates a single 1024-second GWF file containing simulated noise data, using the [ET_10_full_cryo_psd](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/blob/main/src/gwsim/detector/noise_curves/ET_10_full_cryo_psd.txt) sensitivity curve from the [CoBA Science Study](https://iopscience.iop.org/article/10.1088/1475-7516/2023/07/068), sampled at 4096 Hz.

You should see output like:

```
INFO Config mode: quick_start_config.yaml
INFO Configuration loaded and validated: 1 simulators
INFO Created simulation plan from config: 1 batches
INFO Created simulation plan with 1 batches
INFO Validating simulation plan with 1 batches
INFO Simulation plan validation completed successfully
INFO Simulation plan validation passed
INFO Executing simulation plan: 1 batches
INFO Simulation plan validation completed successfully
INFO Executing 1 simulators
Executing simulation plan:   0%|                                        | 0/1
Executing simulation plan: 100%|████████████████████████████████████████| 1/1
INFO All batches completed successfully. Checkpoint files cleaned up.
INFO Simulation completed successfully. Output written to data
```

## 2. Check the Output

Your working directory will contain:

```
output/
└── E-E1-NOISE_STRAIN-1577491218-1024.gwf
metadata/
├── index.yaml
└── noise-0.metadata.yaml
```

### Data File

The GWF file contains the simulated strain data, which can be read using standard gravitational wave analysis software such as [GWpy](https://gwpy.github.io/).
For a quick guide on reading and working with GWF files, see the [Reading Data](reading-data.md) page.

### Metadata File

The metadata file contains everything needed to reproduce this exact simulation.
For a quick guide on how to inspect and reuse metadata files to reproduce a dataset, see the [Metadata Files](metadata.md) page.

## 3. Explore Different Simulators

To run any gwsim simulations, you only need to provide a `.yaml` configuration file.
A collection of ready-to-use configuration files is available in the [`gwsim/examples`](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/tree/main/examples) directory.
You can use them directly or adapt them to suit your needs.

- For an overview of all available examples, see the [Examples](examples.md) page.
- For a more complete and user-friendly guide to writing your own configuration files, see the [Configuration Files](configuration.md) page.

## 4. Next Steps

- [Generating Data](generating-data.md) - Quick guide for generating datasets with gwsim, including Einstein Telescope (ET) examples.
- [Examples](examples.md) - Example configuration files for ET simulations.
- [Request New Features](../dev/contributing.md) - How to request new features or improvements.
- [API Reference](../reference/index.md) - Programmatic usage documentation.
