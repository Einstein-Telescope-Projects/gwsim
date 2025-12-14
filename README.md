# gwsim

[![Pipeline](https://gitlab.et-gw.eu/et-projects/software/gwsim/badges/main/pipeline.svg)](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/pipelines)
[![Documentation](https://app.readthedocs.org/projects/gwsim/badge/?version=latest)](https://gwsim.readthedocs.io/en/latest/)
[![Coverage Report](https://gitlab.et-gw.eu/et-projects/software/gwsim/badges/main/coverage.svg)](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/commits/main)
[![PyPI Version](https://img.shields.io/pypi/v/gwsim)](https://pypi.org/project/gwsim/)
[![Python Versions](https://img.shields.io/pypi/pyversions/gwsim)](https://pypi.org/project/gwsim/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/blob/main/LICENSE)
[![Security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![DOI](https://zenodo.org/badge/1115995501.svg)](https://doi.org/10.5281/zenodo.17925458)

A Python package for generating Mock Data Challenge (MDC) datasets for the gravitational-wave (GW) community. It simulates strain data for detectors like Einstein Telescope, providing a unified interface for reproducible GW data generation.

## Features

-   **Modular Design**: Uses mixins for flexible simulator composition
-   **Detector Support**: Built-in support for various GW detectors with custom configuration options
-   **Waveform Generation**: Integrates with PyCBC and LALSuite for accurate signal simulation
-   **Noise Models**: Supports colored and correlated noise generation (In-Progress)
-   **Population Models**: Handles injection populations for signals and glitches
-   **Data Formats**: Outputs in standard GW formats (GWF frames)
-   **CLI Interface**: Command-line tools for easy simulation workflows

## Installation

We recommend using `uv` to manage virtual environments for installing gwsim.

If you don't have `uv` installed, you can install it with pip. See the project pages for more details:

-   Install via pip: `pip install --upgrade pip && pip install uv`
-   Project pages: [uv on PyPI](https://pypi.org/project/uv/) | [uv on GitHub](https://github.com/astral-sh/uv)
-   Full documentation and usage guide: [uv docs](https://docs.astral.sh/uv/)

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
git clone https://gitlab.et-gw.eu/et-projects/software/gwsim.git
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

-   `globals`: Shared parameters (sampling rate, duration, etc.)
-   `simulators`: List of noise, signal, and glitch generators

## Documentation

Full documentation to be available at readthedocs.io.

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a merge request

## Testing

Run the test suite:

```bash
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions or issues, please open an issue on [GitLab](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/issues/new) or contact the maintainers.
