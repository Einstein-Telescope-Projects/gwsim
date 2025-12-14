# gwsim

A Python package for simulating gravitational-wave detector data for mock data challenges.

## Overview

gwsim provides a unified framework for generating Mock Data Challenge (MDC) datasets for the gravitational-wave community.
It focuses on **usability**, **robustness**, and **extensibility**, to become a standard tool for the community.

### Key Principles

-   **Avoid Reinventing the Wheel**: Builds on established packages (PyCBC, LALSuite, scipy, astropy) for waveform generation and signal processing.
-   **Orchestration Layer**: Provides configuration management, reproducible workflows, and unified interfaces.
-   **Stable CLI Interface**: Remains unchanged regardless of underlying implementation changes.
-   **Extensible**: New signal types can be added without CLI modifications.

## Features

### Signal Simulation

-   **Compact Binary Coalescence (CBC)**: Generates gravitational-wave signals using PyCBC and LALSuite.
-   **Flexible Waveform Models**: Supports a wide range of approximants.
-   **Population Models**: Generates signals from astrophysically realistic populations.

### Noise Simulation

-   **Colored Noise**: Generates noise with a specified power spectral density (PSD).
-   **Correlated Noise**: Produces multi-detector correlated noise using a cross-power spectral density (CSD).
-   **Standard Noise Models**: Integrates PyCBC and Bilby for standard detector noise models.
-   **Glitches**: Injects glitches from realistic populations to simulate transient noise artifacts.

### Data Management

-   **Reproducible Workflows**: Tracks full configuration and state information with checksums.
-   **Safe File Operations**: Writes output atomically, with rollback on failure.
-   **Metadata Tracking**: Stores complete provenance for every generated segment.
-   **Checkpointing**: Resumes interrupted simulations from the last checkpoint.

## Architecture

gwsim uses a **mixin-based architecture**:

-   **Base Simulators** providing core interfaces and state handling
-   Modular **mixins** for detectors, randomness, and time series
-   **Specialized simulators** for noise and signal generation

This enables flexible composition, code reuse, and consistent interfaces.

## Community Focus

gwsim is designed to become a standard tool for MDC generation in the gravitational-wave community by providing:

-   **Integration-Friendly**: Uses thin wrappers around existing tools instead of re-implementing functionality.
-   **Documentation**: Includes extensive examples and API documentation.
-   **Testing**: A comprehensive test suite with high coverage.

## Next Steps

-   [Installation](user-guide/installation.md) - Detailed installation instructions
-   [API Reference](reference/index.md) - Detailed API documentation
-   [Contributing](dev/contributing.md) - How to contribute to the project
-   [Troubleshooting](dev/troubleshooting.md) - Identify and resolve common problems

---

_gwsim is developed for gravitational-wave research._
