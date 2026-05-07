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
        "detector_network": ["ET1"],
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
            "path": "output/signal-0.gwf",
            "channels": ["ET1:STRAIN"],
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

`config` stores the resolved configuration snapshot for that run.
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
and `gwmock-pop`, then rerun the same config with the same seed:

```bash
gwmock simulate config.yaml --seed=42
```

In batch reproduction workflows, the generated `*.metadata.json` files can also
be passed back to `gwmock simulate` directly.
