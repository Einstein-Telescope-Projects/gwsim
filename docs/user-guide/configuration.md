# Configuration Files

This guide explains how to use and write configuration files to generate
datasets tailored to your needs.

## Command-Line Options

### Command [`simulate`](/reference/gwmock/cli/simulate/?h=)

```bash
gwmock simulate config.yaml
```

This is the primary command used to generate mock data. It takes a `.yaml`
configuration file as input, which defines the simulation parameters.

#### Flag `--overwrite` (optional)

By default, gwmock does not overwrite existing output files. If a file already
exists, the tool will raise an error and halt execution. To force overwriting of
existing files, use the `--overwrite` flag:

```bash
gwmock simulate config.yaml --overwrite
```

#### Flag `--dry-run` (optional)

Test your configuration without generating data:

```bash
gwmock simulate config.yaml --dry-run
```

This validates the configuration and shows what would be generated without
actually creating files.

#### Flag `--metadata` (optional)

Generate metadata files along with the data (automatically enabled by default):

```bash
gwmock simulate config.yaml --metadata
```

Metadata files contain complete provenance information including:

- Simulator configuration
- Random number generator state
- Output file names
- Version information

#### Flags `--author` and `--email` (optional)

Include author information in the metadata files:

```bash
gwmock simulate config.yaml --author <your-name> --email <your-email>
```

### Command [`config`](/reference/gwmock/cli/default_config/?h=)

```bash
gwmock config <flag>
```

This command is used to manage default and example configuration files. Exactly
one of the flags `--list`, `--get`, or `--init` must be provided.

#### Flag `--list`

List all the available example configuration files stored in the
[`examples`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples)
directory (see the [Examples](examples.md) page).

```bash
gwmock config --list
```

#### Flag `--get`

Copy one of the available example configuration files from the
[`examples`](https://github.com/Leuven-Gravity-Institute/gwmock/tree/main/examples)
directory into the working directory. The `<example_label>` must be one of the
example names listed by the `gwmock config --list` command.

```bash
gwmock config --get <example_label>
```

#### Flag `--init`

Creates a default configuration file and saves it to the working directory.

```bash
gwmock config --init config.yaml
```

#### Flag `--overwrite` (optional)

By default, gwmock does not overwrite existing configuration files. If a file
already exists, the tool will raise an error and halt execution. To force
overwriting of existing files, use the `--overwrite` flag:

```bash
gwmock config --overwrite
```

#### Flag `--output` (optional)

Specifies the directory where the configuration file will be saved. This flag
must be used together with `--get` or `--init`. If not provided, the working
directory is used by default.

```bash
gwmock config --get <label of the configuration file> --output <directory or file>
```

## Configuration File Structure

The configuration file uses YAML format. They consist of a shared `globals`
section plus exactly one execution surface: legacy `simulators` or the
adapter-backed [orchestration](orchestration.md) schema.

### Globals

Top-level shared parameters used across all simulators:

```yaml
globals:
    working-directory: .
    output-directory: output
    metadata-directory: metadata
    simulator-arguments:
        sampling-frequency:
        duration:
        start-time:
        total-duration:
    output-arguments: {}
```

**Key parameters:**

- `working-directory`: Base directory for operations
- `output-directory`: Where to save generated data files
- `metadata-directory`: Where to save metadata files
- `sampling-frequency`: Sample rate in Hz
- `duration`: Duration of each segment in seconds
- `start-time`: GPS start time
- `total-duration`: Total duration of the dataset
- `output-arguments`: Additional global arguments passed to the file writer

### Simulators

List of simulators to run, each with configuration:

```yaml
simulators:
    noise:
        class: gwmock_noise.ColoredNoiseSimulator
        arguments:
        output:
            file_name:
            arguments:
```

**Simulator properties:**

- `class`: Fully qualified class name of the simulator
- `arguments`: Parameters passed to the simulator
- `output.file_name`: Template for output file naming (supports Jinja2 syntax)
- `output.arguments`: Channel naming and other output metadata

In-tree noise simulator classes have been removed from `gwmock`. For legacy
`simulators.noise.class` configs, point `class` at a public `gwmock_noise.*`
implementation such as `gwmock_noise.ColoredNoiseSimulator` or
`gwmock_noise.CorrelatedNoiseSimulator`. For new configurations, prefer the
adapter-backed `orchestration.noise` flow.

Legacy `signal` simulator classes have been removed. Configure
gravitational-wave signals under `orchestration.signal` instead.

Transient glitches are now configured on the noise side under
`orchestration.noise.arguments.glitches` using public `gwmock-noise` glitch
models. For example:

```yaml
orchestration:
    noise:
        arguments:
            glitches:
                - kind: gengli_blip
                  rate: 0.0011111111111111111
                  amplitude_distribution:
                      distribution: lognormal
                      mean: 1.0
                      std: 0.0
                  population_file: glitches.hdf5
                  psd_file: https://example.org/ET_10_full_cryo_psd.txt
```

## Template Variables

You can use Jinja2-style templates in configuration values such as file names
and channel names:

```yaml
simulators:
    noise:
        class: gwmock_noise.ColoredNoiseSimulator
        arguments:
            detectors:
                - E1_Triangle_EMR
                - E2_Triangle_EMR
                - E3_Triangle_EMR
        output:
            file_name:
                'E-{{ detectors }}_NOISE_STRAIN-{{ start_time }}-{{ duration
                }}.gwf'
            arguments:
                channel: '{{ detectors }}:STRAIN'
```

In this example, `file_name` and `channel` are automatically updated for each
detector being processed.

**Common variables:**

- `{{ start_time }}`: GPS start time from globals
- `{{ duration }}`: Segment duration from globals
- `{{ detectors }}`: Current detector being processed

## Checkpointing

gwmock automatically creates checkpoints during long simulations. If a process
is interrupted:

1. A `.gwmock_checkpoint/simulation.checkpoint.json` file is saved in the
   working directory
2. Rerun the same command to resume from the last checkpoint
3. The tool automatically detects and continues from where it left off

```bash
# Start simulation
gwmock simulate config.yaml

# If interrupted (Ctrl+C, crash, etc.), resume with same command
gwmock simulate config.yaml
```

The checkpoint contains:

- Simulator state
- Progress information
- Already-generated file tracking

## Best Practices

1. **Use templates**: Leverage Jinja2 templates for dynamic configuration
2. **Set seeds**: Always set `seed` for reproducibility
3. **Check space**: Ensure sufficient disk space before long runs
4. **Use dry-run**: Test configurations with `--dry-run` before full simulation
5. **Organize outputs**: Use descriptive `output-directory` and
   `metadata-directory` names
