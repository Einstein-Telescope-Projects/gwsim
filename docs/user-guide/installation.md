# Installation

We recommend using `uv` to manage virtual environments for installing gwmock.

If you don't have `uv` installed, you can install it with pip. See the project
pages for more details:

- Install via pip: `pip install --upgrade pip && pip install uv`
- Project pages: [uv on PyPI](https://pypi.org/project/uv/) |
  [uv on GitHub](https://github.com/astral-sh/uv)
- Full documentation and usage guide: [uv docs](https://docs.astral.sh/uv/)

## Requirements

- Python 3.12 or 3.13
- Operating System: Linux, macOS, or Windows

<!-- prettier-ignore -->
!!!note
    The package is built and tested against Python 3.12-3.13. When creating a virtual environment with `uv`,
    specify the Python version to ensure compatibility: `uv venv --python 3.12` (replace `3.12` with your
    preferred version in the 3.12-3.13 range). This avoids potential issues with unsupported Python versions.

## Install from PyPI

The recommended way to install gwmock is from PyPI:

```bash
# Create a virtual environment (recommended with uv)
uv venv gwmock-env --python 3.12
source gwmock-env/bin/activate  # On Windows: gwmock-env\Scripts\activate
uv pip install gwmock
```

### Optional Dependencies

For development or specific features:

```bash
# Development dependencies (testing, linting, etc.)
uv pip install gwmock[dev]

# Documentation dependencies
uv pip install gwmock[docs]

# All dependencies
uv pip install gwmock[dev,docs]
```

## Install from Source

For the latest development version:

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock.git
ce gwmock
# Create a virtual environment (recommended with uv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install .
```

### Development Installation

To set up for development:

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock.git
cd gwmock

# Create a virtual environment (recommended with uv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install ".[dev]"

# Install the commitlint dependencies
npm install

# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg
```

## Verify Installation

Check that gwmock is installed correctly:

```bash
gwmock --help
```

```bash
python -c "import gwmock; print(gwmock.__version__)"
```

## Dependencies

### Subpackage adoption baseline

Later refactor issues may assume this dependency baseline without reopening the
packaging question:

- Python `>=3.12,<3.14`
- `gwmock-signal>=0.5.0`
- `gwmock-noise>=0.1.2`
- `gwmock-pop>=0.6.0`

The baseline also depends on two upstream assumptions that are now explicit:

- `gwmock-pop` ISS-026 exports `CBC_PARAMETER_NAMES` for downstream validation
  and cross-package contract tests.
- `gwmock-signal` ISS-014 made the public license surface internally consistent;
  if that license declaration changes upstream, revisit downstream adoption
  before broadening the dependency plan.

### Core Dependencies

- **typer**: CLI framework
- **numpy**: Numerical computing
- **pycbc**: Gravitational-wave data analysis
- **h5py**: HDF5 file format support
- **pydantic**: Data validation
- **tqdm**: Progress bars

### Optional Dependencies

- **gengli**: Glitch generation (for transient artifacts)

## Getting Help

1. Check the [troubleshooting guide](../dev/troubleshooting.md)
2. Search existing
   [issues](https://github.com/Leuven-Gravity-Institute/gwmock/issues)
3. Create a new issue with:
    - Your operating system and Python version
    - Full error message
    - Steps to reproduce the problem
