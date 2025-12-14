# Troubleshooting

## Version Conflicts During Installation

If you encounter version conflicts or dependency errors while installing `gwsim`, this is often due to conflicting package versions in your Python environment. To resolve this:

1. **Use a Virtual Environment**: Always install `gwsim` in a dedicated virtual environment to isolate it from your system Python and other projects. We recommend using `uv` for this:

    ```bash
    uv venv --python 3.10 gwsim-env  # Replace 3.10 with your preferred version (3.10-3.12)
    source gwsim-env/bin/activate  # On Windows: gwsim-env\Scripts\activate
    uv pip install gwsim
    ```

2. **Check Python Version**: Ensure you're using Python 3.10, 3.11, or 3.12, as these are the versions the package is built and tested against. Using an unsupported version (e.g., 3.9 or 3.13+) may cause compatibility issues:

    ```bash
    python --version  # Should show 3.10.x, 3.11.x, or 3.12.x
    ```

3. **Update Dependencies**: If conflicts persist, try updating `pip` and `uv`:

    ```bash
    pip install --upgrade pip uv
    uv pip install --upgrade gwsim
    ```

4. **Clean Install**: If issues continue, remove the virtual environment and recreate it:
    ```bash
    rm -rf gwsim-env  # Or delete the folder
    uv venv --python 3.10 gwsim-env
    source gwsim-env/bin/activate
    uv pip install gwsim
    ```

For more details on installation, see the [Installation Guide](../user-guide/installation.md). If problems persist, check the [GitLab issues](https://gitlab.et-gw.eu/et-projects/software/gwsim/-/issues) or create a new issue with your Python version and full error output.
