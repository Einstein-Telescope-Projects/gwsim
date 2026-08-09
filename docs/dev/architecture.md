# Architecture

This document describes the high-level architecture and design principles of
gwmock.

## Overview

gwmock is designed as an orchestration layer that leverages existing third-party
packages for the physics layer. The package provides:

- **Configuration Management**: YAML-based configuration with inheritance and
  template expansion
- **Reproducible Workflows**: Full state tracking with checksums and metadata
- **Protocol-backed Extensibility**: Third-party backends plug in through public
  protocols and backend resolution
- **Adapter-backed Layout**: In-tree `signal/`, `noise/`, and `population/`
  packages expose adapters only

## Core Design Principles

### 1. Avoid Reinventing the Wheel

gwmock wraps existing, battle-tested libraries rather than reimplementing signal
processing algorithms. This approach:

- Ensures correctness by relying on established implementations
- Reduces maintenance burden
- Allows users to leverage decades of gravitational-wave research

**Key dependencies:**

- **gwmock-signal**: public signal protocol and adapter surface
- **gwmock-noise**: public noise protocol and adapter surface
- **gwmock-pop**: public population protocol and adapter surface
- **typer** and **pydantic**: CLI and configuration plumbing

### 2. Stable CLI Interface

The command-line interface remains unchanged regardless of backend changes. New
behavior is added by updating adapters, orchestration, and the public protocols,
not by adding physics implementations to `gwmock`.

### 3. Orchestration Helpers (`simulator/`, `utils/`)

The package keeps shared orchestration helpers for deterministic seeds, state
tracking, and output layout. These helpers support the adapters, but they are
not physics implementations.

Benefits:

- Deterministic orchestration
- Clean separation between adapters and backend physics
- Centralized checkpoint and seed handling
- Simple to extend without changing the CLI surface

## Project Structure

```text
gwmock/
├── __init__.py
├── __main__.py
├── cli/
│   ├── __init__.py
│   ├── main.py              # Typer CLI entry point
│   ├── simulate.py          # Simulation command
│   ├── simulate_utils.py
│   ├── validate.py          # Output-vs-metadata hash validation command
│   ├── batch.py             # Batch helpers
│   ├── merge.py             # Merge helpers
│   ├── config.py            # Configuration command (fetch/interactive editor)
│   ├── adapter_orchestration.py
│   ├── repository/          # Zenodo commands (create/upload/publish/…)
│   └── utils/               # Config loading & resolution, templating,
│                            # backend resolver, checkpointing, metadata,
│                            # simulation plan, hashing, retry
├── simulator/
│   ├── __init__.py
│   ├── base.py              # Base Simulator class
│   ├── state.py             # StateAttribute descriptor and checkpoint state helpers
│   └── seeds.py             # Deterministic seed derivation
├── signal/
│   ├── __init__.py
│   └── adapter.py           # Signal adapter
├── noise/
│   ├── __init__.py
│   └── adapter.py           # Noise adapter
├── population/
│   ├── __init__.py
│   └── adapter.py           # Population adapter
├── data/
│   ├── __init__.py
│   ├── serialize/           # Encoder/decoder for serializable objects
│   └── time_series/         # Time-series containers and signal injection
├── mixin/
│   ├── __init__.py
│   ├── randomness.py        # Randomness mixin for simulators
│   └── time_series.py       # Time-series mixin for simulators
├── monitor/
│   ├── __init__.py
│   └── resource.py          # Resource monitoring helpers
├── repository/
│   ├── __init__.py
│   └── zenodo.py            # Zenodo REST client
├── utils/
│   ├── __init__.py
│   ├── io.py                # File I/O utilities
│   ├── log.py               # Logging setup
│   ├── random.py            # Random number management
│   ├── download.py          # Download helpers
│   ├── retry.py             # Retry helpers
│   ├── datetime_parser.py   # Human-friendly duration/date parsing
│   ├── population.py        # Population utilities
│   ├── et_2l_geometry.py    # ET 2L detector geometry
│   └── triangular_et_geometry.py  # Triangular ET detector geometry
└── version.py               # Version information
```

## Key Components

### 1. CLI Layer (`cli/`)

**Purpose**: User-facing command-line interface

**Key files:**

- `main.py`: Typer application with commands
- `simulate.py`: Main simulation command
- `utils/`: Configuration loading, checkpointing, templating

**Features:**

- Commands: `gwmock simulate config.yaml`
- Flags: `--overwrite`, `--dry-run`, `--metadata`
- Argument validation and help text

### 2. Simulator Framework (`simulator/`)

**Purpose**: Core simulator interface and registration

**Key classes:**

- `Simulator`: Abstract base with state management
- `StateAttribute`: Descriptor for state tracking
- `PopulationIterationState`: Legacy population checkpoint state for
  orchestration resume

### 4. Adapter Layer (`signal/`, `noise/`, `population/`)

**Purpose**: Translate orchestration configs into the public subpackage
protocols.

These packages do not contain physics implementations. They resolve public
backends, validate conformance, and hand off to the relevant subpackage or
third-party class.

### 5. Backend Integration

**Purpose**: Third-party backend support through the public contracts.

Backends may be shipped by `gwmock`, discovered through entry points, or
referenced directly as `module:Class`.

### 6. Configuration System (`cli/utils/config.py`)

**Features:**

- YAML parsing and validation
- Jinja2 template expansion
- Configuration inheritance
- Runtime variable substitution

**Example flow:**

```text
config.yaml (user input)
    ↓
YAML parsing
    ↓
Inheritance resolution (if inherits field present)
    ↓
Template expansion (Jinja2)
    ↓
    Backend resolution
    ↓
Validated SimulationPlan
```

### 7. Checkpointing (`cli/utils/checkpoint.py`)

**Purpose**: Resume interrupted simulations

**Checkpoint structure:**

```json
{
  "last_completed_batch": 5,
  "last_completed_file": "file.hdf5",
  "random_state": {...},
  "processed_samples": 5,
  "timestamp": "2025-01-01T12:00:00Z"
}
```

**Resume logic:**

1. Load checkpoint file
2. Restore random state
3. Skip completed batches
4. Continue from last incomplete batch

### 8. State Management (`simulator/state.py`)

**Purpose**: Track simulator state across batches

**StateAttribute descriptor:**

```python
class StateAttribute:
    """Descriptor for state tracking without class-level pollution."""

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj._state.get(self.name)

    def __set__(self, obj, value):
        obj._state[self.name] = value
```

**Key feature**: Instance-level state isolation prevents cross-contamination in
tests

## Data Flow

### Simulation Workflow

```text
User Input (config.yaml)
    ↓
CLI parsing (Typer)
    ↓
Configuration Loading
    - Parse YAML
    - Resolve inheritance
    - Expand templates
    ↓
Validation
    - Check file paths
    - Validate classes
    - Verify parameters
    ↓
SimulationPlan creation
    ↓
Checkpoint check
    - Load if exists
    - Skip completed batches
    ↓
Simulator instantiation
    - Resolve class from registry
    - Inject configuration
    ↓
Batch iteration
    ├── Generate data
    ├── Create time series
    ├── Write GWF file
    ├── Generate metadata
    └── Update checkpoint
    ↓
Output
    - Data files (*.hdf5, *.gwf)
    - Metadata files (*.metadata.json)
    - Checkpoint file (.gwmock_checkpoint/simulation.checkpoint.json)
```

### Data Generation

```text
Adapter.resolve()
    ↓
Public protocol backend
    ↓
Generated strain or population data
    ↓
Adapter output formatting
    ↓
gwf file + metadata
```

## Extension Points

### Adding a Third-Party Backend

1. **Implement the upstream protocol** in your package.
2. **Expose the class** through an entry point or importable `module:Class`
   reference.
3. **Reference it in config**:

    ```yaml
    orchestration:
        noise:
            backend: my_package.noise:MyCustomNoise
            arguments:
                param1: value1
    ```

### Adding an Orchestration Helper

1. **Create the helper** in `simulator/` or `utils/`:

    ```python
    class MyHelper:
        """Provides orchestration-only functionality."""

        def my_method(self):
            pass
    ```

2. **Use it from an adapter or CLI helper**, not from a physics package.

## Thread Safety & Concurrency

**Current implementation:**

- Single-threaded batch processing
- Checkpointing ensures fault tolerance
- Random state management prevents seed collisions

**Future considerations:**

- Thread-pool execution for batch parallelization
- Process-pool for computationally intensive simulations
- Distributed simulation across multiple machines

## Testing Strategy

### Unit Tests

- Mock third-party libraries
- Test configuration parsing
- Test state management
- Test CLI argument handling

### Integration Tests

- End-to-end simulation workflows
- Checkpoint/resume functionality
- File I/O operations

### Performance Tests

- Benchmark common operations
- Memory profiling for large datasets
- Stress testing with extended simulations

## Design Decisions

### Why Mixins?

- **Flexibility**: Combine features as needed
- **Reusability**: Same mixin in multiple simulators
- **Maintainability**: Changes in one mixin don't affect others
- **Testability**: Easy to mock individual mixins

### Why StateAttribute?

- **Instance isolation**: Prevents test interference
- **Clean interface**: Transparent to users
- **Automatic tracking**: Integrated with checkpointing

### Why Registry?

- **Dynamic loading**: Simulators added without code changes
- **Configuration-driven**: Full control via YAML
- **Third-party integration**: Easy to wrap external libraries
- **Discovery**: Automatic detection of available simulators

## Performance Considerations

1. **Lazy loading**: Simulators instantiated only when needed
2. **Streaming**: Process data in chunks to reduce memory
3. **Caching**: Cache compiled templates and registry lookups
4. **Checkpointing**: Resume from intermediate states
5. **Parallelization**: Process multiple batches concurrently

## References

- [PyCBC Documentation](https://pycbc.org/)
- [LALSuite Documentation](https://lscsoft.docs.ligo.org/lalsuite/)
- [GWpy Documentation](https://gwpy.github.io/)
- [Bilby Documentation](https://bilby.readthedocs.io/)
