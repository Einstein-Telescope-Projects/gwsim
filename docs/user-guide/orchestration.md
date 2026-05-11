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

The `orchestration:` section requires all three sub-sections — `population`,
`signal`, and `noise` — to be present. None of them are optional.

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

For protocol details and third-party backend integration, see
[Protocol Contracts](protocols.md) and [Extensibility](extensibility.md).
