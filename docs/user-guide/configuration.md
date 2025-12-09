# Configuration Files

This guide explains how to use and write configuration files to generate datasets tailored to your needs.

## Command-Line Options

### Command [`simulate`](/reference/gwsim/cli/simulate/?h=)

```bash
gwsim simulate config.yaml
```
This is the primary command used to generate mock data.
It takes a `.yaml` configuration file as input, which defines the simulation parameters.

#### Flag `--overwrite` (optional)

By default, gwsim does not overwrite existing output files. If a file already exists, the tool will raise an error and halt execution.
To force overwriting of existing files, use the `--overwrite` flag:

```bash
gwsim simulate config.yaml --overwrite
```

#### Flag `--dry-run` (optional)

Test your configuration without generating data:


```bash
gwsim simulate config.yaml --dry-run
```

This validates the configuration and shows what would be generated without actually creating files.

#### Flag `--metadata` (optional)

Generate metadata files along with the data (automatically enabled by default):

```bash
gwsim simulate config.yaml --metadata
```

Metadata files contain complete provenance information including:

- Simulator configuration
- Random number generator state
- Output file names
- Version information

#### Flags `--author` and `--email` (optional)

Include author information in the metadata files:

```bash
gwsim simulate config.yaml --author <your-name> --email <your-email>
```

### Command [`config`](/reference/gwsim/cli/default_config/?h=)

```bash
gwsim config <flag>
```

This command is used to manage default and example configuration files.
Exactly one of the flags `--list`, `--get`, or `--init` must be provided.

#### Flag `--list`

List all the available example configuration files stored in the [`examples`](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/tree/main/examples?ref_type=heads) directory (see the [Examples](examples.md) page).

```bash
gwsim config --list
```

#### Flag `--get`

Copy one of the available example configuration files from the [`examples`](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/tree/main/examples?ref_type=heads) directory into the working directory.

```bash
gwsim config --get <config.yaml>
```

#### Flag `--init`

Creates a default configuration file and saves it to the working directory.

```bash
gwsim config --init
```

#### Flag `--overwrite` (optional)

By default, gwsim does not overwrite existing configuration files. If a file already exists, the tool will raise an error and halt execution.
To force overwriting of existing files, use the `--overwrite` flag:

```bash
gwsim config --overwrite
```

#### Flag `--output` (optional)


Specifies the directory where the configuration file will be saved.
This flag must be used together with `--get` or `--init`.
If not provided, the working directory is used by default.

```bash
gwsim config --output <directory>
```

## Configuration File Structure

The configuration file uses YAML format. They consist of two main sections: globals and simulators.

### Globals

Top-level shared parameters used across all simulators:

```yaml
globals:
  working-directory:
  sampling-frequency:
  duration:
  total-duration:
  start-time:
  output-directory:
  metadata-directory:
  seed-base:
```

**Key parameters:**

- `working-directory`: Base directory for operations
- `sampling-frequency`: Sample rate in Hz
- `duration`: Duration of each segment in seconds
- `total-duration`: Total duration of the dataset
- `start-time`: GPS start time
- `output-directory`: Where to save generated data files
- `metadata-directory`: Where to save metadata files
- `seed-base`: Base random seed for reproducibility

### Simulators

List of simulators to run, each with configuration:

```yaml
simulators:
  noise:
    class:
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

For details on simulator-specific `arguments`, refer to the [API Reference](../reference/index.md) page.

Available `noise` simulators includes:

- [`BaseNoise`](/reference/gwsim/noise/base/#gwsim.noise.base.NoiseSimulator)
- [`WhiteNoiseSimulator`](/reference/gwsim/noise/white_noise/?h=whitenoise#gwsim.noise.white_noise.WhiteNoiseSimulator)
- [`ColoredNoiseSimulator`](/reference/gwsim/noise/colored_noise/?h=colorednoise#gwsim.noise.colored_noise.ColoredNoiseSimulator)
- [`CorrelatedNoiseSimulator`](/reference/gwsim/noise/correlated_noise/?h=correlatednoise#gwsim.noise.correlated_noise.CorrelatedNoiseSimulator)

Available `signal` simulators includes:

- [`CBCSignalSimulator`](/reference/gwsim/signal/cbc/?h=gwsim.signal.cbc.cbcsignalsimulator#gwsim.signal.cbc.CBCSignalSimulator)

Available `glitch` simulators includes:

- [`GengliGlitchSimulator`](/reference/gwsim/glitch/gengli_glitch/?h=gwsim.glitch.gengli_glitch.gengliglitchsimulator#gwsim.glitch.gengli_glitch.GengliGlitchSimulator)

## Template Variables

You can use Jinja2-style templates in configuration values such as file names and channel names:

```yaml
simulators:
  noise:
    arguments:
        detectors:
          - E1_Triangle_EMR
          - E2_Triangle_EMR
          - E3_Triangle_EMR
    output:
      file_name: "E-{{ detectors }}_NOISE_STRAIN-1000000000-1024.gwf"
      arguments:
        channel: "{{ detectors }}:STRAIN"
```

In this example, `file_name` and `channel` are automatically updated for each detector being processed.

**Common variables:**

- `{{ start_time }}`: GPS start time from globals
- `{{ duration }}`: Segment duration from globals
- `{{ detectors }}`: Current detector being processed
- `{{ seed_base }}`: Random seed base from globals

## Checkpointing

gwsim automatically creates checkpoints during long simulations. If a process is interrupted:

1. A `checkpoint.json` file is saved in the working directory
2. Rerun the same command to resume from the last checkpoint
3. The tool automatically detects and continues from where it left off

```bash
# Start simulation
gwsim simulate config.yaml

# If interrupted (Ctrl+C, crash, etc.), resume with same command
gwsim simulate config.yaml
```

The checkpoint contains:

- Simulator state
- Progress information
- Already-generated file tracking

## Best Practices

1. **Use templates**: Leverage Jinja2 templates for dynamic configuration
2. **Set seeds**: Always set `seed-base` for reproducibility
3. **Check space**: Ensure sufficient disk space before long runs
4. **Use dry-run**: Test configurations with `--dry-run` before full simulation
5. **Organize outputs**: Use descriptive `output-directory` and `metadata-directory` names
