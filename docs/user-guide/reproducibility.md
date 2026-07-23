# Reproducibility

`gwmock` writes one versioned JSON provenance record per generated batch as
`*.metadata.json`.

## Schema

Each record is validated at write time and uses schema version `1.0.0`.
Consumers must reject unknown major versions.

```json
{
    "schema_version": "1.0.0",
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
            "path": "output/signal/E-ET1_SARD_STRAIN_BBH-1577491218-1024.gwf",
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

For the config shape that feeds this record, see
[Orchestration](orchestration.md) and [Protocol Contracts](protocols.md).

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
