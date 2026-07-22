# Design Walkthrough

This page explains how `gwmock` is designed and why, at the level a reader needs
before opening the source code. For the quick summary, see the
[Package Walkthrough](walkthrough.md); for developer-level detail, see the
[Architecture](../dev/architecture.md) page.

## The core idea: orchestration, not physics

`gwmock` deliberately contains **no physics implementations**. Waveforms, noise
models, and populations live in three split-out packages, each consumed through
a stable public protocol:

| Physics domain | Package                                                                    | Protocol                                |
| -------------- | -------------------------------------------------------------------------- | --------------------------------------- |
| Populations    | [gwmock-pop](https://github.com/Leuven-Gravity-Institute/gwmock-pop)       | `GWPopSimulator`                        |
| Signals        | [gwmock-signal](https://github.com/Leuven-Gravity-Institute/gwmock-signal) | `GWSimulator` (`simulate(...)` surface) |
| Noise          | [gwmock-noise](https://github.com/Leuven-Gravity-Institute/gwmock-noise)   | `NoiseSimulator` + `open_stream(...)`   |

This buys three properties:

1. **Correctness by delegation** — the physics packages wrap established
   libraries (e.g. PyCBC/LALSuite waveform models) rather than reimplementing
   them, and can be validated independently of the orchestration.
2. **A stable CLI** — new physics arrives by upgrading or swapping backends,
   never by changing the `gwmock` command surface.
3. **Extensibility without forking** — any third-party class satisfying a
   protocol can be referenced directly from the YAML config
   ([Extensibility](extensibility.md), [Protocol Contracts](protocols.md)).

Inside `gwmock`, the `population/`, `signal/`, and `noise/` packages are
**adapters only**: they resolve the configured backend, validate protocol
conformance, invoke it, and format the outputs. If a backend does not satisfy
its protocol, configuration fails clearly before any data is generated.

## Life of a simulation

What happens when you run `gwmock simulate config.yaml`:

```text
config.yaml
    │  parse YAML, resolve inheritance, expand Jinja2 templates
    ▼
Validated SimulationPlan          (cli/utils/config*.py, simulation_plan.py)
    │  resolve backends: alias → entry point → module:Class
    ▼
Adapters                          (population/, signal/, noise/)
    │  derive per-batch deterministic seeds        (simulator/seeds.py)
    │  check for a checkpoint, skip completed work (cli/utils/checkpoint.py)
    ▼
Batch loop
    ├─ sample population → generate signals → generate noise
    ├─ write GWF frames                        (data/, utils/io.py)
    ├─ write .metadata.json provenance record  (cli/utils/metadata.py)
    └─ update checkpoint
    ▼
output/*.gwf  +  metadata/*.metadata.json  +  resource_usage_summary.json
```

Three orchestration mechanisms are worth understanding in the source:

- **Configuration resolution** — YAML configs support inheritance and Jinja2
  templating, and are validated into a typed plan with pydantic before anything
  runs. See [Configuration Files](configuration.md) and
  `src/gwmock/cli/utils/config_resolution.py`.
- **Deterministic randomness** — every batch's seed is derived from the
  top-level seed, so runs are reproducible and batches are independent
  (`src/gwmock/simulator/seeds.py`, `src/gwmock/utils/random.py`).
- **Checkpoint/resume** — long runs record progress and random state, and can
  resume after interruption without repeating or corrupting completed batches
  (`src/gwmock/cli/utils/checkpoint.py`, `src/gwmock/simulator/state.py`).

## Reproducibility as a first-class output

Every run writes a `.metadata.json` file recording the fully resolved
configuration, all seeds, the versions of `gwmock` and its physics backends, and
the SHA-256 hash of every output file. This single file supports three
workflows:

- **Reproduce** — `gwmock simulate run.metadata.json` regenerates the dataset
  bit-for-bit ([Reproducibility](reproducibility.md)).
- **Verify** — `gwmock validate` re-hashes outputs and compares against the
  recorded values ([Validating Output Files](validate.md)).
- **Publish** — `gwmock repository` uploads datasets with their provenance to
  Zenodo ([Publishing Data to Zenodo](repository.md)).

This is the dataset's chain of custody: a published frame file can be traced
back to the exact configuration, seed, and package versions that produced it,
and checked for integrity at every step.

## Source tree

```text
src/gwmock/
├── cli/                    # Typer CLI — start reading here
│   ├── main.py             #   entry point, command registration
│   ├── simulate.py         #   the simulate command
│   ├── validate.py         #   output-vs-metadata hash validation
│   ├── batch.py, merge.py  #   split/merge helpers for large runs
│   ├── repository/         #   Zenodo commands (create/upload/publish/…)
│   └── utils/              #   config loading & resolution, templating,
│                           #   backend resolver, checkpoint, simulation plan
├── population/adapter.py   # adapters: orchestration config → protocol calls
├── signal/adapter.py       #   (no physics in any of these)
├── noise/adapter.py
├── simulator/              # base simulator, state tracking, seed derivation
├── data/                   # time-series containers, injection, serialization
├── mixin/                  # randomness & time-series mixins for simulators
├── utils/                  # io, logging, random state, download,
│                           #   ET geometries (triangular & 2L)
├── repository/zenodo.py    # Zenodo REST client
├── monitor/resource.py     # resource-usage monitoring
└── version.py
```

Full API documentation for every module is in the
[API Reference](../reference/index.md); tests mirror this layout under `tests/`.

## Suggested reading order for the source code

1. `cli/main.py` → `cli/simulate.py` — how a run starts.
2. `cli/utils/simulation_plan.py` and `cli/utils/config_resolution.py` — what a
   validated plan contains.
3. `cli/utils/backend_resolver.py` — how `backend:` strings become classes.
4. One adapter end-to-end, e.g. `noise/adapter.py`, together with the
   [Protocol Contracts](protocols.md) page.
5. `simulator/seeds.py` and `cli/utils/checkpoint.py` — determinism and resume,
   the parts reproducibility rests on.
6. `cli/utils/metadata.py` and `cli/validate.py` — provenance and integrity.
