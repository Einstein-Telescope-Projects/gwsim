# gwsim

A Python package for simulating gravitational wave detector data for mock data challenges.

## Overview

gwsim provides a unified framework for generating Mock Data Challenge (MDC) datasets for the gravitational wave community. It emphasizes **usability**, **robustness**, and **extensibility** to become a standard tool for the community.

### Key Principles

- **Avoid Reinventing the Wheel**: Leverages existing third-party packages (PyCBC, LALSuite, scipy, astropy) for actual signal processing and waveform generation.
- **Orchestration Layer**: Provides configuration management, reproducible workflows, and unified interfaces.
- **Stable CLI Interface**: Remains unchanged regardless of underlying implementation changes.
- **Extensible**: New signal types can be added without CLI modifications.

## Features

### Signal Simulation
- **Compact Binary Coalescence (CBC)**: Generate gravitational wave signals using PyCBC and LALSuite.
- **Flexible Waveform Models**: Support for multiple approximants.
- **Population Models**: Generate signals from astrophysically realistic populations.

### Noise Simulation
- **Colored Noise**: Generate noise with specified power spectral density (PSD).
- **Correlated Noise**: Multi-detector correlated noise using cross-spectral density (CSD).
- **Standard Noise Models**: PyCBC and Bilby integration for standard detector noise models.
- **Glitches**: Injection of glitches from realistic populations for transient noise artifacts.

### Data Management
- **Reproducible Workflows**: Full configuration and state tracking with checksums.
- **Safe File Operations**: Safe file writing with transaction-like rollbacks.
- **Metadata Tracking**: Complete provenance information for each generated segment.
- **Checkpointing**: Resume interrupted simulations from last checkpoint.

## Architecture

The package uses a **mixin-based composition** pattern for maximum flexibility:

- **Base Simulator**: Core interface with state management and iteration capabilities.
- **Mixins**: Modular functionality (RandomnessMixin, DetectorMixin, TimeSeriesMixin, etc.).
- **Specialized Simulators**: Combine base + mixins for specific use cases (NoiseSimulator, SignalSimulator).

This design allows:

- Easy extension with new simulator types.
- Consistent interfaces across different simulators.
- Code reuse and maintainability.

## Community Standard

gwsim is designed to become the standard tool for MDC generation in the gravitational wave community by:

- **Production-Ready**: Thread-safety, comprehensive logging, graceful error handling.
- **Integration-Friendly**: Thin wrappers around existing tools rather than reimplementation.
- **Documentation**: Extensive examples and API documentation.
- **Testing**: Comprehensive test suite with high coverage.

## Next Steps

- [Installation Guide](user-guide/installation.md) - Detailed installation instructions
- [Configuration Guide](user-guide/configuration.md) - Complete configuration reference
- [Examples](user-guide/examples.md) - Real-world usage examples
- [API Reference](reference/index.md) - Detailed API documentation
- [Contributing](dev/contributing.md) - How to contribute to the project

---

*gwsim is developed for gravitational wave research.*
