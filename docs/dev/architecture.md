# Architecture

This document describes the high-level architecture and design principles of gwsim.

## Overview

gwsim is designed as an orchestration layer that leverages existing third-party packages (PyCBC, LALSuite, scipy, astropy) for actual signal processing and waveform generation. The package provides:

- **Configuration Management**: YAML-based configuration with inheritance and template expansion
- **Reproducible Workflows**: Full state tracking with checksums and metadata
- **Unified Interfaces**: Consistent APIs across different simulator types
- **Extensibility**: Easy addition of new simulators without CLI modifications

## Core Design Principles

### 1. Avoid Reinventing the Wheel

gwsim wraps existing, battle-tested libraries rather than reimplementing signal processing algorithms. This approach:
- Ensures correctness by relying on established implementations
- Reduces maintenance burden
- Allows users to leverage decades of gravitational-wave research

**Key dependencies:**

- **PyCBC**: Waveform generation and data analysis
- **LALSuite**: LAL algorithms for GW science
- **scipy**: Scientific computing utilities
- **astropy**: Astronomical utilities and units

### 2. Stable CLI Interface

The command-line interface remains unchanged regardless of underlying implementation changes. New features are added through:

- New simulator classes in `signal/`, `noise/`, `glitch/` modules
- Updated configuration options
- No CLI modifications required

### 3. Mixin-Based Composition

The package uses a **mixin pattern** for maximum flexibility and code reuse:

```
Base Simulator (interface, state management, iteration)
    ↓
    ├── + RandomnessMixin → handles random number generation
    ├── + DetectorMixin → handles detector-specific operations
    ├── + TimeSeriesMixin → handles time series data
    ├── + PopulationReaderMixin → handles population file I/O
    └── + WaveformMixin → handles waveform generation

    ↓
Specialized Simulators (NoiseSimulator, SignalSimulator, GlitchSimulator)
```

Benefits:

- Code reuse across simulator types
- Clean separation of concerns
- Easy to combine functionality
- Simple to extend with new mixins

## Project Structure

```
gwsim/
├── __init__.py
├── cli/
│   ├── __init__.py
│   ├── main.py              # Typer CLI entry point
│   ├── simulate.py          # Simulation command
│   ├── config.py            # Configuration utilities
│   ├── default_config.py    # Default configuration
│   └── utils/               # CLI utilities
│       ├── checkpoint.py    # Checkpointing logic
│       ├── config.py        # Config loading/validation
│       ├── retry.py         # Retry decorators
│       ├── simulation_plan.py # Simulation planning
│       ├── template.py      # Template expansion
│       └── utils.py         # Helper utilities
├── simulator/
│   ├── __init__.py
│   ├── base.py              # Base Simulator class
│   ├── state.py             # StateAttribute descriptor
│   ├── registry.py          # Simulator registry
│   ├── noise.py             # Noise simulator base
│   ├── signal.py            # Signal simulator base
│   └── glitch.py            # Glitch simulator base
├── mixin/
│   ├── __init__.py
│   ├── detector.py          # DetectorMixin
│   ├── randomness.py        # RandomnessMixin
│   ├── time_series.py       # TimeSeriesMixin
│   ├── waveform.py          # WaveformMixin
│   ├── population_reader.py # PopulationReaderMixin
│   └── gwf.py               # GWF frame handling
├── noise/
│   ├── __init__.py
│   ├── base.py              # BaseNoise class
│   ├── colored.py           # Colored noise
│   └── correlated.py        # Correlated noise
├── signal/
│   ├── __init__.py
│   ├── cbc.py               # CBC waveforms
│   └── population/          # Population models
├── glitch/
│   ├── __init__.py
│   ├── base.py              # BaseGlitch class
│   └── gengli.py            # Gengli glitch generation
├── population/
│   ├── __init__.py
│   ├── cbc.py               # CBC population models
│   └── glitch.py            # Glitch population models
├── detector/
│   ├── __init__.py
│   ├── detectors/           # Detector configurations
│   └── noise_curves/        # PSD files
├── data/
│   ├── __init__.py
│   └── ...                  # Data utilities
├── utils/
│   ├── __init__.py
│   ├── io.py                # File I/O utilities
│   ├── log.py               # Logging setup
│   ├── random.py            # Random number management
│   └── validation.py        # Configuration validation
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

- Commands: `gwsim simulate config.yaml`
- Flags: `--overwrite`, `--dry-run`, `--metadata`
- Argument validation and help text

### 2. Simulator Framework (`simulator/`)

**Purpose**: Core simulator interface and registration

**Key classes:**

- `Simulator`: Abstract base with state management
- `StateAttribute`: Descriptor for state tracking

### 3. Mixin System (`mixin/`)

**Purpose**: Modular functionality for simulators

**Key mixins:**

- `RandomnessMixin`: Seeded RNG management
- `DetectorMixin`: Multi-detector support
- `TimeSeriesMixin`: Time series handling
- `WaveformMixin`: Waveform generation
- `PopulationReaderMixin`: Population file reading

**Mixin pattern example:**
```python
class NoiseSimulator(BaseSimulator, RandomnessMixin, DetectorMixin, TimeSeriesMixin):
    """Noise simulator with randomness, detector, and time series support."""
    pass
```

### 4. Signal Generators (`signal/`, `noise/`, `glitch/`)

**Purpose**: Specific simulator implementations

**Thin wrappers around third-party libraries:**

```python
class PyCBCStationaryGaussianNoiseSimulator(NoiseSimulator):
    """Generate stationary Gaussian noise using PyCBC."""

    def generate(self, **params):
        # Delegate to PyCBC
        return pycbc.waveform.noise.gaussian_noise(
            psd_name=self.psd,
            duration=self.duration,
            # ...
        )
```

### 5. Configuration System (`cli/utils/config.py`)

**Features:**

- YAML parsing and validation
- Jinja2 template expansion
- Configuration inheritance
- Runtime variable substitution

**Example flow:**
```
config.yaml (user input)
    ↓
YAML parsing
    ↓
Inheritance resolution (if inherits field present)
    ↓
Template expansion (Jinja2)
    ↓
Class registry resolution
    ↓
Validated SimulationPlan
```

### 6. Checkpointing (`cli/utils/checkpoint.py`)

**Purpose**: Resume interrupted simulations

**Checkpoint structure:**
```json
{
  "last_completed_batch": 5,
  "last_completed_file": "file.gwf",
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

### 7. State Management (`simulator/state.py`)

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

**Key feature**: Instance-level state isolation prevents cross-contamination in tests

## Data Flow

### Simulation Workflow

```
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
    - Data files (*.gwf)
    - Metadata files (*.yaml)
    - Checkpoint file (checkpoint.json)
```

### Data Generation

```
Simulator.generate()
    ↓
RandomnessMixin (seed management)
    ↓
Third-party library (PyCBC, LALSuite, etc.)
    ↓
Raw signal data
    ↓
TimeSeriesMixin (format to gwpy.TimeSeries)
    ↓
DetectorMixin (apply detector response)
    ↓
GWFMixin (write to frame file)
    ↓
gwf file + metadata
```

## Extension Points

### Adding a New Noise Simulator

1. **Create new class** in `noise/`:
```python
class MyCustomNoise(BaseNoise, RandomnessMixin, TimeSeriesMixin):
    def generate(self, **params):
        # Implementation
        pass
```

2. **Register in CLI** (automatic via entry points or manual in registry)

3. **Use in config**:
```yaml
simulators:
  my_noise:
    class: gwsim.noise.MyCustomNoise
    arguments:
      param1: value1
```

### Adding a New Mixin

1. **Create mixin class**:
```python
class MyMixin:
    """Provides custom functionality."""

    def my_method(self):
        pass
```

2. **Use in simulator**:
```python
class MySimulator(BaseSimulator, MyMixin):
    pass
```

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
