
# Colored Noise Simulator Example

Brief example showing how to run the colored-noise simulator included with gwsim.

- **Location**: `examples/noise/colored_noise_simulator`
- **Config**: uses the local `config.yaml` in this example directory (edit to change parameters)
- **Purpose**: demonstrate generating detector-colored noise, per-detector output files, and metadata sidecars.

Quick steps
- Create a virtualenv and install dev requirements (project root):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

- Run the example (from this directory):

```bash
make all
```

Or manually:

```bash
gwsim simulate config.yaml
```

- Clean up generated files:

```bash
make clean
```

What to expect
- One or more GWF output files in the `output/` directory (one per detector when configured).
- A metadata YAML sidecar per batch in `metadata/` (with external .npy files for large arrays).

Notes
- File and channel names may use `{{ ... }}` template variables (e.g. `{{ detectors }}:STRAIN`).
- Use `total_duration` in the config to generate multiple batches; `max_samples` is a fallback.
