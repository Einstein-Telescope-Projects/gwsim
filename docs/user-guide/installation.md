# Installation

## Requirements

- Python 3.10 or higher
- Operating System: Linux, macOS, or Windows

## Install from PyPI

The recommended way to install gwsim is from PyPI:

```bash
pip install gwsim
```

### Optional Dependencies

For development or specific features:

```bash
# Development dependencies (testing, linting, etc.)
pip install gwsim[dev]

# Documentation dependencies
pip install gwsim[docs]

# All dependencies
pip install gwsim[dev,docs]
```

## Install from Source

For the latest development version:

```bash
git clone https://gitlab.et-gw.eu/et-projects/software/gwsim.git
cd gwsim
pip install -e .
```

### Development Installation

To set up for development:

```bash
git clone https://gitlab.et-gw.eu/et-projects/software/gwsim.git
cd gwsim
pip install -e .[dev]
pre-commit install  # Set up pre-commit hooks
```

## Verify Installation

Check that gwsim is installed correctly:

```bash
gwsim --help
```

```bash
python -c "import gwsim; print(gwsim.__version__)"
```

## Dependencies

### Core Dependencies

- **typer**: CLI framework
- **numpy**: Numerical computing
- **pycbc**: Gravitational wave data analysis
- **bilby**: Gravitational wave data utilities
- **h5py**: HDF5 file format support
- **pydantic**: Data validation
- **tqdm**: Progress bars

### Optional Dependencies

- **gengli**: Glitch generation (for transient artifacts)

## Troubleshooting

### Common Issues

**Permission Errors**
```bash
# Use user installation
pip install --user gwsim
```

**Version Conflicts**



```bash
# Create a virtual environment (recommended with uv)
uv venv gwsim-env
source gwsim-env/bin/activate  # On Windows: gwsim-env\Scripts\activate
uv pip install gwsim

# Or with standard venv
python -m venv gwsim-env
source gwsim-env/bin/activate  # On Windows: gwsim-env\Scripts\activate
pip install gwsim
```

If you don't have `uv` installed, you can install it with pip. See the project pages for more details:

- Install via pip: `pip install --upgrade pip && pip install uv`
- Project pages: [uv on PyPI](https://pypi.org/project/uv/) | [uv on GitHub](https://github.com/astral-sh/uv)
- Full documentation and usage guide: [uv docs](https://docs.astral.sh/uv/)

### Getting Help

1. Check the [troubleshooting guide](../dev/troubleshooting.md)
2. Search existing [issues](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/issues)
3. Create a new issue with:
    - Your operating system and Python version
    - Full error message
    - Steps to reproduce the problem
