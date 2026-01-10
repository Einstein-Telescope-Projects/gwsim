# Running Simulations on a Cluster

The `gwsim batch` command allows you to build and submit gwsim simulations as batch jobs on a Slurm-based cluster.

## Overview

The `gwsim batch` command has two mutually exclusive modes:

1. **Create a batch-ready configuration file** from one of the provided examples.
   This mode is triggered by the `--get` option.

2. **Generate a Slurm submit script** (and optionally submit the job) from an existing configuration file that already contains a `batch` section.

## 1. Create a Batch-ready Configuration File

Use this mode when starting from an example configuration file in the in the [`examples/`](https://github.com/Leuven-Gravity-Institute/gwsim/tree/main/examples) directory and you want to prepare a configuration file that includes all necessary information for batch submission.

```bash
gwsim batch --get <example_label> [options]
```

This command requires the label of the example configuration file to copy, which can be obtained using the `gwsim config --list` command.
It copies `examples/<example_label>/config.yaml` and adds a complete `batch` section (see the [Examples](examples.md) page).

The following default resources are always added:

```yaml
nodes: 1
ntasks-per-node: 1
cpus-per-task: 1
mem: 16GB
```

<!-- prettier-ignore -->
!!! note
    gwsim currently does not support multi-threaded execution.
    To modify the memory request, edit the configuration file manually.

### Commonly used options (only allowed with `--get`)

- `--job-name <name>`
  Job name that will appear in SLURM (stored as `batch.job-name`). Default: `gwsim_job`.

- `--scheduler <scheduler>`
  Name of the scheduler (only `slurm` currently supported). Default: `slurm`.

- `--account <account>`
  SLURM account/project to charge.

- `--cluster <partition>`
  SLURM cluster or partition to run on.

- `--time <time>`
  Wall time limit in `hh:mm:ss` format.

- `--extra-line '<command>'`
  Add a custom shell line to the submit script before the simulation command (e.g. environment setup, module loads, conda activate).
  Can be repeated multiple times.

- `--output <path>`
  Destination for the new configuration file.
  Default: `config.yaml` in the current directory.

- `--overwrite`
  Overwrite the output configuration file if it already exists.

### Example

The following command:

```bash
gwsim batch --get default_config \
  --job-name gwsim_test \
  --account my_account \
  --cluster cluster_name \
  --time 02:00:00 \
  --extra-line 'export PATH="/my_account/miniconda3/bin:$PATH"' \
  --extra-line 'eval "$(conda shell.bash hook)"' \
  --extra-line 'conda activate /my_account/miniconda3/envs/my_env'
```

add the following `batch` section to the configuration file:

```yaml
batch:
  scheduler: slurm # Default
  job-name: gwsim_test
  resources:
    nodes: 1 # Default
    ntasks-per-node: 1 # Default
    cpus-per-task: 1 # Default
    mem: 16GB # Default
  submit:
    account: my_account
    cluster: cluster_name
    time: 02:00:00
  extra_lines:
    - export PATH="/my_account/miniconda3/bin:$PATH"
    - eval "$(conda shell.bash hook)"
    - conda activate /my_account/miniconda3/envs/my_env
```

## 2. Generate and Submit a Slurm Job

Use this mode when you already have a configuration file that contains a valid `batch` section.

```bash
gwsim batch <config.yaml> [--submit]
```

This command requires the path to a configuration file that contains a `batch` section with at least `scheduler` and `job-name` (default resources are assumed).
When executed, the following actions are performed:

1. Directories are created under `<working-directory>/slurm/`:

   - `output/` – stdout files
   - `error/` – stderr files
   - `submit/` – the generated `.submit` script

2. A SLURM submit script is written containing:

   - All `#SBATCH` directives from `batch.resources`
   - Any additional `#SBATCH` directives from `batch.submit` (account, cluster, time, etc.)
   - All custom lines from `batch.extra_lines` (if present)
   - The command `gwsim simulate <absolute_path_to_config.yaml>`

3. If `--submit` is used, `sbatch` is called.

### Optional

- `--submit`
  Immediately submit the generated job using `sbatch`.
  Without this flag, only the submit script is created.

- `--overwrite`
  Overwrite an existing submit script if it already exists.

### Example

```bash
# Just generate the submit script and save in `<working-directory>/slurm/submit`
gwsim batch config.yaml

# Generate and submit immediately
gwsim batch config.yaml --submit
```
