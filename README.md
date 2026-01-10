# gwsim

[![Python CI](https://github.com/Leuven-Gravity-Institute/gwsim/actions/workflows/CI.yml/badge.svg)](https://github.com/Leuven-Gravity-Institute/gwsim/actions/workflows/CI.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Leuven-Gravity-Institute/gwsim/main.svg)](https://results.pre-commit.ci/latest/github/Leuven-Gravity-Institute/gwsim/main)
[![Documentation Status](https://github.com/Leuven-Gravity-Institute/gwsim/actions/workflows/documentation.yml/badge.svg)](https://leuven-gravity-institute.github.io/gwsim)
[![codecov](https://codecov.io/gh/Leuven-Gravity-Institute/gwsim/graph/badge.svg?token=GLW2LEFKW7)](https://codecov.io/gh/Leuven-Gravity-Institute/gwsim)
[![PyPI Version](https://img.shields.io/pypi/v/gwsim)](https://pypi.org/project/gwsim/)
[![Python Versions](https://img.shields.io/pypi/pyversions/gwsim)](https://pypi.org/project/gwsim/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Leuven-Gravity-Institute/gwsim/blob/main/LICENSE)
[![Security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![DOI](https://zenodo.org/badge/1115995501.svg)](https://doi.org/10.5281/zenodo.17925458)

A Python package for generating Mock Data Challenge (MDC) datasets for the gravitational-wave (GW) community. It simulates strain data for detectors like Einstein Telescope, providing a unified interface for reproducible GW data generation.

## Features

- **Modular Design**: Uses mixins for flexible simulator composition
- **Detector Support**: Built-in support for various GW detectors with custom configuration options
- **Waveform Generation**: Integrates with PyCBC and LALSuite for accurate signal simulation
- **Noise Models**: Supports colored and correlated noise generation (In-Progress)
- **Population Models**: Handles injection populations for signals and glitches
- **Data Formats**: Outputs in standard GW formats (GWF frames)
- **CLI**: Command-line tools for easy simulation workflows

## Installation

We recommend using `uv` to manage virtual environments for installing gwsim.

If you don't have `uv` installed, you can install it with pip. See the project pages for more details:

- Install via pip: `pip install --upgrade pip && pip install uv`
- Project pages: [uv on PyPI](https://pypi.org/project/uv/) | [uv on GitHub](https://github.com/astral-sh/uv)
- Full documentation and usage guide: [uv docs](https://docs.astral.sh/uv/)

**Note:** The package is built and tested against Python 3.10-3.12. When creating a virtual environment with `uv`, specify the Python version to ensure compatibility: `uv venv --python 3.10` (replace `3.10` with your preferred version in the 3.10-3.12 range). This avoids potential issues with unsupported Python versions.

### From PyPI

```bash
# Create a virtual environment (recommended with uv)
uv venv gwsim-env --python 3.10
source gwsim-env/bin/activate  # On Windows: gwsim-env\Scripts\activate
uv pip install gwsim
```

### From Source

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwsim.git
ce gwsim
# Create a virtual environment (recommended with uv)
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install .
```

## Quick Start

### Command Line

```bash
# Generate simulated data
gwsim simulate config.yaml
```

## Configuration

gwsim uses YAML configuration files for reproducible simulations. See `examples/config.yaml` for a complete example.

Key configuration sections:

- `globals`: Shared parameters (sampling rate, duration, etc.)
- `simulators`: List of noise, signal, and glitch generators

## Documentation

Full documentation to be available at [https://leuven-gravity-institute.github.io/gwsim](https://leuven-gravity-institute.github.io/gwsim).

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a merge request

### Release Schedule

Releases follow a fixed schedule: every Tuesday at 00:00 UTC,
unless an emergent bugfix is required.
This ensures predictable updates while allowing flexibility for critical issues.
Users can view upcoming changes in the draft release on the
[GitHub Releases page](https://github.com/Leuven-Gravity-Institute/gwsim/releases).

## Testing

Run the test suite:

```bash
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions or issues, please open an issue on [GitHub](https://github.com/Leuven-Gravity-Institute/gwsim/issues/new) or contact the maintainers.
