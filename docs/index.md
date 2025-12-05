# gwsim

A Python package for simulating gravitational-wave detector data for mock data challenges.

## Overview

gwsim provides a unified framework for generating Mock Data Challenge (MDC) datasets for the gravitational-wave community.
It focuses on **usability**, **robustness**, and **extensibility**, to become a standard tool for the community.

### Key Principles

- **Avoid Reinventing the Wheel**: Builds on established packages (PyCBC, LALSuite, scipy, astropy) for waveform generation and signal processing.
- **Orchestration Layer**: Provides configuration management, reproducible workflows, and unified interfaces.
- **Stable CLI Interface**: Remains unchanged regardless of underlying implementation changes.
- **Extensible**: New signal types can be added without CLI modifications.

## Features

### Signal Simulation
- **Compact Binary Coalescence (CBC)**: Generates gravitational-wave signals using PyCBC and LALSuite.
- **Flexible Waveform Models**: Supports a wide range of approximants.
- **Population Models**: Generates signals from astrophysically realistic populations.

### Noise Simulation
- **Colored Noise**: Generates noise with a specified power spectral density (PSD).
- **Correlated Noise**: Produces multi-detector correlated noise using a cross-power spectral density (CSD).
- **Standard Noise Models**: Integrates PyCBC and Bilby for standard detector noise models.
- **Glitches**: Injects glitches from realistic populations to simulate transient noise artifacts.

### Data Management
- **Reproducible Workflows**: Tracks full configuration and state information with checksums.
- **Safe File Operations**: Writes output atomically, with rollback on failure.
- **Metadata Tracking**: Stores complete provenance for every generated segment.
- **Checkpointing**: Resumes interrupted simulations from the last checkpoint.

## Architecture

The gwsim package uses a **mixin-based composition** pattern for maximum flexibility:

- **Base Simulator**: Core interface with state management and iteration capabilities.
- **Mixins**: Modular functionality (RandomnessMixin, DetectorMixin, TimeSeriesMixin, etc.).
- **Specialized Simulators**: Combine base + mixins for specific use cases (NoiseSimulator, SignalSimulator).

This design allows:

- Easy extension with new simulator types.
- Consistent interfaces across all simulators.
- Code reuse and maintainability.

## Community Standard

gwsim is designed to become a standard tool for MDC generation in the gravitational-wave community by providing:

- **Production-Ready**: Thread-safety, comprehensive logging, graceful error handling.
- **Integration-Friendly**: Uses thin wrappers around existing tools instead of re-implementing functionality.
- **Documentation**: Includes extensive examples and API documentation.
- **Testing**: A comprehensive test suite with high coverage.

## Next Steps

- [Installation](user-guide/installation.md) - Detailed installation instructions
- [API Reference](reference/index.md) - Detailed API documentation
- [Contributing](dev/contributing.md) - How to contribute to the project
- [Troubleshooting](dev/troubleshooting.md) - Identify and resolve common problems

---

*gwsim is developed for gravitational-wave research.*
