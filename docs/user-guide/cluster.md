# Running Simulations on a Cluster

The `gwmock batch` command allows you to build and submit gwmock simulations as
batch jobs on a cluster running either **Slurm** or **HTCondor**.

## Overview

The `gwmock batch` command has two mutually exclusive modes:

1. **Create a batch-ready configuration file** from one of the provided
   examples. This mode is triggered by the `--get` option.

2. **Generate a scheduler submit file** (and optionally submit the job) from an
   existing configuration file that already contains a `batch` section. The
   scheduler — `slurm` or `htcondor` — is read from `batch.scheduler`.

## 1. Create a Batch-ready Configuration File

Use this mode when starting from an example configuration file in the
[`examples/`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples)
directory and you want to prepare a configuration file that includes all
necessary information for batch submission.

```bash
gwmock batch --get <example_label> [options]
```

This command requires the label of the example configuration file to copy, which
can be obtained using the `gwmock config --list` command. It copies
`examples/<example_label>/config.yaml` and adds a complete `batch` section (see
the [Examples](examples.md) page).

Default resources are always added. The keys are **scheduler-native** — they are
emitted verbatim into the submit file — so the defaults differ per scheduler:

=== "Slurm"

    ```yaml
    nodes: 1
    ntasks-per-node: 1
    cpus-per-task: 1
    mem: 16GB
    ```

=== "HTCondor"

    ```yaml
    request_cpus: 1
    request_memory: 16GB
    request_disk: 4GB
    ```

<!-- prettier-ignore -->
!!! note
    gwmock currently does not support multi-threaded execution.
    To modify the memory request, edit the configuration file manually.

### Commonly used options (only allowed with `--get`)

- `--scheduler <scheduler>` Name of the scheduler: `slurm` or `htcondor`.
  Default: `slurm`.

- `--job-name <name>` Job name used in the generated submit file and output file
  names (stored as `batch.job-name`). Default: `gwmock_job`.

- `--account <account>` Account/project to charge (stored in `batch.submit`).

- `--cluster <partition>` Cluster or partition to run on (stored in
  `batch.submit`).

- `--time <time>` Wall time limit in `hh:mm:ss` format (stored in
  `batch.submit`).

- `--extra-line '<command>'` Add a custom shell line to run before the
  simulation command (e.g. environment setup, module loads, conda activate). Can
  be repeated multiple times.

- `--output <path>` Destination for the new configuration file. Default:
  `config.yaml` in the current directory.

- `--overwrite` Overwrite the output configuration file if it already exists.

<!-- prettier-ignore -->
!!! warning "Scheduler-native keys in `batch.submit`"
    Entries under `batch.submit` are written verbatim into the submit file:
    as `#SBATCH --<key>=<value>` directives for Slurm, and as `<key> = <value>`
    submit commands for HTCondor. The `--account`, `--cluster`, and `--time`
    convenience options map to Slurm's `sbatch` options; for HTCondor, edit
    `batch.submit` in the configuration file and use native submit commands
    (e.g. `accounting_group`) instead.

### Example

The following command:

```bash
gwmock batch --get default_config \
  --job-name gwmock_test \
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
    job-name: gwmock_test
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

An equivalent HTCondor `batch` section looks like:

```yaml
batch:
    scheduler: htcondor
    job-name: gwmock_test
    resources:
        request_cpus: 1 # Default
        request_memory: 16GB # Default
        request_disk: 4GB # Default
    submit:
        accounting_group: my_account
    extra_lines:
        - export PATH="/my_account/miniconda3/bin:$PATH"
        - eval "$(conda shell.bash hook)"
        - conda activate /my_account/miniconda3/envs/my_env
```

## 2. Generate and Submit a Job

Use this mode when you already have a configuration file that contains a valid
`batch` section.

```bash
gwmock batch <config.yaml> [--submit]
```

This command requires the path to a configuration file that contains a `batch`
section with at least `scheduler` and `job-name` (default resources are
assumed). When executed, the following actions are performed:

1. Directories are created under `<working-directory>/<scheduler>/` (i.e.
   `slurm/` or `htcondor/`):
    - `output/` – stdout files
    - `error/` – stderr files
    - `submit/` – the generated submit file(s)
    - `log/` – HTCondor job event logs (HTCondor only)

2. The scheduler-specific submit file is written to `submit/`:

    === "Slurm"

        A single `sbatch` script `<job-name>.submit` containing:

        - All `#SBATCH` directives from `batch.resources`
        - Any additional `#SBATCH` directives from `batch.submit` (account,
          cluster, time, etc.)
        - All custom lines from `batch.extra_lines` (if present)
        - The command `gwmock simulate <absolute_path_to_config.yaml>`

    === "HTCondor"

        Two files:

        - `<job-name>.sub` — the submit description file (`universe = vanilla`),
          containing the `output`/`error`/`log` paths, all entries from
          `batch.resources` and `batch.submit` as submit commands, and a
          `queue` statement.
        - `<job-name>.sh` — an executable wrapper script that the job runs on
          the execute node. HTCondor submit files cannot carry shell setup, so
          all custom lines from `batch.extra_lines` and the command
          `gwmock simulate <absolute_path_to_config.yaml>` live here.

3. If `--submit` is used, the job is submitted with the scheduler's native
   command: `sbatch` for Slurm, `condor_submit` for HTCondor.

### Optional

- `--submit` Immediately submit the generated job using `sbatch` or
  `condor_submit`. Without this flag, only the submit file(s) are created.

- `--overwrite` Overwrite an existing submit file (and, for HTCondor, wrapper
  script) if it already exists.

### Example

```bash
# Just generate the submit file(s) in `<working-directory>/<scheduler>/submit`
gwmock batch config.yaml

# Generate and submit immediately
gwmock batch config.yaml --submit
```
