# Installation

We recommend using `uv` to manage virtual environments for installing gwsim.

If you don't have `uv` installed, you can install it with pip. See the project pages for more details:

-   Install via pip: `pip install --upgrade pip && pip install uv`
-   Project pages: [uv on PyPI](https://pypi.org/project/uv/) | [uv on GitHub](https://github.com/astral-sh/uv)
-   Full documentation and usage guide: [uv docs](https://docs.astral.sh/uv/)

## Requirements

-   Python 3.10 or higher
-   Operating System: Linux, macOS, or Windows

<!-- prettier-ignore -->
!!!note
    The package is built and tested against Python 3.10-3.12. When creating a virtual environment with `uv`, specify the Python version to ensure compatibility: `uv venv --python 3.10` (replace `3.10` with your preferred version in the 3.10-3.12 range). This avoids potential issues with unsupported Python versions.

## Install from PyPI

The recommended way to install gwsim is from PyPI:

```bash
# Create a virtual environment (recommended with uv)
uv venv gwsim-env --python 3.10
source gwsim-env/bin/activate  # On Windows: gwsim-env\Scripts\activate
uv pip install gwsim
```

### Optional Dependencies

For development or specific features:

```bash
# Development dependencies (testing, linting, etc.)
uv pip install gwsim[dev]

# Documentation dependencies
uv pip install gwsim[docs]

# All dependencies
uv pip install gwsim[dev,docs]
```

## Install from Source

For the latest development version:

```bash
git clone https://gitlab.et-gw.eu/et-projects/software/gwsim.git
ce gwsim
# Create a virtual environment (recommended with uv)
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install .
```

### Development Installation

To set up for development:

```bash
git clone https://gitlab.et-gw.eu/et-projects/software/gwsim.git
ce gwsim

# Create a virtual environment (recommended with uv)
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install ".[dev]"

# Install the commitlint dependencies
npm install

# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg
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

-   **typer**: CLI framework
-   **numpy**: Numerical computing
-   **pycbc**: Gravitational-wave data analysis
-   **bilby**: Gravitational-wave data utilities
-   **h5py**: HDF5 file format support
-   **pydantic**: Data validation
-   **tqdm**: Progress bars

### Optional Dependencies

-   **gengli**: Glitch generation (for transient artifacts)

## Getting Help

1. Check the [troubleshooting guide](../dev/troubleshooting.md)
2. Search existing [issues](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/issues)
3. Create a new issue with:
    - Your operating system and Python version
    - Full error message
    - Steps to reproduce the problem
