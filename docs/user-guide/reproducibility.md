# Reproducibility

`gwmock` writes one versioned JSON provenance record per generated batch as
`*.metadata.json`.

## Schema

Each record is validated at write time and uses schema version `1.5.0`.
Consumers must reject unknown major versions.

```json
{
    "schema_version": "1.5.0",
    "gwmock_version": "x.y.z",
    "subpackage_versions": {
        "gwmock_signal": "x.y.z",
        "gwmock_noise": "x.y.z",
        "gwmock_pop": "x.y.z"
    },
    "config": {},
    "resolved_config": {},
    "replayable": true,
    "config_sha256": "...",
    "seed": 42,
    "segment_seeds": [123456789, 987654321],
    "population": {
        "backend": "module:Class",
        "source_type": "bbh",
        "n_events": 128,
        "parameter_names": [],
        "metadata": {}
    },
    "signal": {
        "backend": "module:Class",
        "waveform_model": "IMRPhenomXPHM",
        "detector_network": ["ET1_SARD", "ET2_SARD", "ET3_SARD"],
        "injections": [
            {
                "event_id": 0,
                "parameters": { "mass_1": 30.0, "coa_time": 1577491218.5 }
            }
        ],
        "metadata": {}
    },
    "noise": {
        "backend": "module:Class",
        "psd": "ET_10_full_cryo_psd",
        "metadata": {}
    },
    "outputs": [
        {
            "kind": "signal",
            "path": "output/signal/E-ET1_SARD_STRAIN_BBH-1577491218-1024.hdf5",
            "channels": ["ET1_SARD:STRAIN"],
            "t0": 1577491218,
            "duration": 1024,
            "sha256": "..."
        }
    ],
    "host": {
        "platform": "...",
        "python": "3.12.x",
        "cpu": "...",
        "git_sha": "..."
    },
    "environment": {
        "python": "3.12.5",
        "python_implementation": "CPython",
        "packages": { "numpy": "2.0.0", "gwmock": "x.y.z", "...": "..." }
    }
}
```

`config` stores the input configuration snapshot for that run (template
variables expanded). `resolved_config` stores the same config with every
runtime-resolved external value folded in — for example a `DeepExtractorGlitch`
whose dataset was downloaded at the repository default is recorded here pinned
to the concrete Hugging Face commit it actually used. It is `null` when nothing
needed resolving (a purely parametric run). **Replay prefers `resolved_config`
over `config`**, so a run that did not explicitly pin its external inputs still
reproduces the exact resources it used.

`replayable` is `true` unless a declared external-mutable input could not be
pinned to an immutable version (e.g. an offline dataset with no local cache); a
`false` run is not bit-for-bit reproducible from its metadata, and replaying it
emits a warning.

`segment_seeds` stores the deterministic per-segment seeds that `gwmock` derives
locally. Adapter-backed noise now consumes one shared
`gwmock_noise.open_stream(...)` iterator per run, so the top-level `seed` is
recorded once and noise continuation no longer appears as one derived seed per
batch. The subpackage `metadata` objects are preserved as JSON objects without
gwmock rewriting their internal structure.

`signal.injections` records the source parameters of the signals attributed to
that batch's frame(s), in injection order. Each entry is
`{"event_id": <index in the population>, "parameters": {...}}`. A signal is
listed against **every frame its samples reach**, so a long inspiral crossing a
segment boundary appears in each frame it spans, and a continuous wave appears
in all of them. This changed in schema 1.5.0: a signal used to be listed only
under the frame it was generated for, which for a 48 s inspiral across 32 s
segments meant one frame out of three -- and not the one holding the merger. The
frame a signal is _generated_ for is the one its waveform **starts** in (schema
1.4.0; before that, the one its `coa_time` fell in). `event_id` is the event's
index in the population as ordered for the run (by `coa_time` under the default
ordering), so it is stable for a fixed configuration. Stationary/SGWB segments
have no discrete events and record an empty list.

`environment` is a full freeze of the environment that produced the run — the
Python version and the version of every installed distribution (direct and
transitive) — recorded so the run can be reproduced against exactly those
dependencies. It is `null` for records written before this field existed.

For the config shape that feeds this record, see
[Orchestration](orchestration.md) and [Protocol Contracts](protocols.md).

## Exact-dependency reproduction (`--isolate`)

By default, reproducing from metadata runs in your current environment and warns
if the recorded package versions differ. For bit-for-bit reproduction against
the exact dependencies of the original run, add `--isolate`:

```bash
gwmock simulate metadata/ --isolate
```

This reads the recorded `environment` freeze, builds a cached, isolated
[uv](https://docs.astral.sh/uv/) virtualenv pinned to those versions (matching
the recorded Python `major.minor`), and re-runs the reproduction inside it. If
the current environment already matches, it runs in place; if no environment was
recorded (older metadata), it warns and runs in place.

Requirements and limits:

- `uv` must be installed, and the recorded package versions must be resolvable
  from your package index — a run made with editable/dev installs (versions not
  published to an index) cannot be recreated this way and will fail loudly
  rather than run in the wrong environment.
- Environments are cached under `~/.cache/gwmock/reproduction-envs` (override
  with `GWMOCK_ENV_CACHE`) and keyed by the version set, so repeated
  reproductions of the same run skip reinstalling.
- **Earth-orientation data has a shelf life, and it is not pinned by a package
  version.** Anything using sidereal time — every projection with
  `earth-rotation: true` — depends on the IERS table Astropy loads. Pinning
  `astropy-iers-data` recreates the table _bundled_ with that release, but
  Astropy's `iers.conf.auto_download` is `True` by default and its
  `auto_max_age` is 30 days: once the pinned release is older than that, Astropy
  fetches the current table from the IERS server instead, and no recorded
  package version captures which one it got. A run reproduced within a month of
  its dependencies' release matches; reproduced a year later, the sidereal time
  can differ.

    To pin it properly, set `iers.conf.auto_download = False` before simulating,
    so the packaged table is used regardless of age — at the cost of using
    Earth-orientation data as old as the pinned release. Measured scale: one
    weekly table release moved a strain peak by 1.6e-06 relative.

## Finding which frame contains a signal

Alongside the per-file `index.yaml`, a run writes `signal_index.yaml` mapping
each signal's `event_id` to the frame file(s) that contain it. The entry records
one contribution per batch, because a signal reaching several segments is
written by several batches; `find-signal` flattens them, and reports `metadata`
as a **list** of batch metadata files on both lookup paths. An index written
before schema 1.5.0 is still read. Use `gwmock find-signal` to resolve a signal
to its frame:

```bash
# By id (fast path via signal_index.yaml)
gwmock find-signal --metadata-dir metadata/ --id 42

# By parameter filters (scans the recorded injections); combine with AND
gwmock find-signal --metadata-dir metadata/ --param mass_1>=30 --param coa_time<1577491300

# Machine-readable
gwmock find-signal --metadata-dir metadata/ --id 42 --json
```

Filters accept `==`, `!=`, `>`, `<`, `>=`, `<=`; numeric values are compared
numerically. The command exits non-zero when no signal matches.

## Reproducing a run

For deterministic reproduction, pin `gwmock`, `gwmock-signal`, `gwmock-noise`,
and `gwmock-pop` to the same versions used originally, then rerun the same
config file. The seed is stored in the config itself:

```bash
gwmock simulate config.yaml
```

In batch reproduction workflows, pass the generated `*.metadata.json` files
directly to `gwmock simulate`. Each metadata file carries the exact config
snapshot and per-segment seeds needed to reproduce that batch independently.
Because replay reads `resolved_config`, any downloaded dataset (e.g. a
DeepExtractor glitch dataset) is refetched at the exact version the original run
used, even if the config never pinned it and the upstream dataset has since
moved:

```bash
# Reproduce specific batches from their metadata files
gwmock simulate metadata/orchestration-0.metadata.json metadata/orchestration-1.metadata.json

# Or reproduce everything from a metadata directory
gwmock simulate metadata/
```
