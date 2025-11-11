# gwsim

[![Pipeline](https://gitlab.et-gw.eu/et-projects/software/gwsim/badges/main/pipeline.svg)](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/pipelines)
[![Coverage](https://gitlab.et-gw.eu/et-projects/software/gwsim/badges/main/coverage.svg)](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/graphs/main/charts)
[![PyPI Version](https://img.shields.io/pypi/v/gwsim)](https://pypi.org/project/gwsim/)
[![Python Versions](https://img.shields.io/pypi/pyversions/gwsim)](https://pypi.org/project/gwsim/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/blob/main/LICENSE)

A Python package for generating Mock Data Challenge (MDC) datasets for the gravitational-wave (GW) community. It simulates strain data for detectors like Einstein Telescope, providing a unified interface for reproducible GW data generation.

## Features

- **Modular Design**: Uses mixins for flexible simulator composition
- **Detector Support**: Built-in support for various GW detectors with custom configuration options
- **Waveform Generation**: Integrates with PyCBC and LALSuite for accurate signal simulation
- **Noise Models**: Supports colored and correlated noise generation (In-Progress)
- **Population Models**: Handles injection populations for signals and glitches
- **Data Formats**: Outputs in standard GW formats (GWF frames)
- **CLI Interface**: Command-line tools for easy simulation workflows

## Installation

### From Source

```bash
git clone https://gitlab.et-gw.eu/et-projects/software/gwsim.git
cd gwsim
pip install -e .
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

Full documentation to be available at readthedocs.io.

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Testing

Run the test suite:

```bash
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions or issues, please open an issue on [GitLab](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/issues/new) or contact the maintainers.
