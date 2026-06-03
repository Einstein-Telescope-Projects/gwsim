# Orchestration

`gwmock` now expects adapter-backed `orchestration:` configs for new runs. This
surface replaces the legacy `simulators:` layout and splits configuration into
three explicit sections:

- `orchestration.population`
- `orchestration.signal`
- `orchestration.noise`

## Migration from `simulators:`

The legacy `simulators:` schema has been removed. Configs that use a top-level
`simulators:` key are now rejected at load time with a clear error message. All
configurations must use the `orchestration:` schema described below. For the
full configuration shape and examples, see the
[Configuration Files](configuration.md) guide.

The `orchestration:` section may contain `population`, `signal`, `noise`, or a
combination of them. CBC-style transient signal generation uses `population`
plus `signal`. Stationary SGWB generation can use `signal` without `population`
by setting `signal.source-type: sgwb`.

```yaml
orchestration:
    population:
        backend: file
        n-samples: 1
        arguments:
            path: population.h5
    signal:
        detectors:
            - H1
        output:
            file_name: signal-{{ counter }}.gwf
            arguments:
                channel: H1:STRAIN
    noise:
        arguments:
            seed: 7
        output:
            file_name: noise-{{ counter }}.npy
```

Signal-only SGWB example:

```yaml
orchestration:
    signal:
        source-type: sgwb
        detectors:
            - ET-Triangle-Sardinia
        minimum-frequency: 5
        parameters:
            omega_ref: 1.0e-9
            spectral_index: 0.0
            reference_frequency: 25.0
        output:
            file_name: sgwb-{{ counter }}.hdf5
```

For protocol details and third-party backend integration, see
[Protocol Contracts](protocols.md) and [Extensibility](extensibility.md).
