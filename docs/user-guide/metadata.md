# Metadata Files

A metadata file captures all information related to a specific simulation run.
It is designed to provide full provenance, traceability, and exact reproducibility.

Each metadata file includes:

- **Author information**

  - Author name
  - Contact email (if provided)

- **File integrity information**

  - File hashes used to uniquely identify outputs and to detect changes

- **Configuration snapshots**

  - Global configuration options from the configuration file
  - Simulator-specific configuration options from the configuration file

- **Simulation context**

  - Simulator name
  - Versions of all packages used during the simulation

- **Execution state**

  - Pre-batch state, describing the state before execution (e.g., random seeds, initial conditions)

- **Output tracking**

  - List of generated output files

- **Timing information**
  - Timestamp marking the start of the simulation

## File Format and Organization

Metadata files use the YAML format and are identified by the `.metadata.yaml` suffix.

Each metadata file stores the complete provenance for a single generated frame.
During dataset generation, an accompanying `index.yaml` file is created to record the association between each generated data file and its corresponding metadata file.

The `index.yaml` file has the following structure:

```yaml
data_file_0.gwf: data_0.metadata.yaml
data_file_1.gwf: data_1.metadata.yaml
data_file_2.gwf: data_2.metadata.yaml
data_file_3.gwf: data_3.metadata.yaml
```

## Reproducing a Dataset from a Metadata File

Metadata files can be used to reproduce an identical dataset.
Each metadata file contains the exact configuration and pre-batch state required for exact reproducibility.

Users may distribute individual metadata files, allowing others to independently reproduce specific simulation batches without access to the original dataset.

To reproduce a dataset from a metadata file, run:

```bash
gwsim simulate metadata_filename.metadata.yaml
```
