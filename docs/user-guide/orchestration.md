# Orchestration

`gwmock` now expects adapter-backed `orchestration:` configs for new runs. This
surface replaces the legacy `simulators:` layout and splits configuration into
three explicit sections:

- `orchestration.population`
- `orchestration.signal`
- `orchestration.noise`

## Migration from `simulators:`

If you still have a legacy config, it will continue to run for now, but the CLI
warns that it will be removed in `v0.5.0`. For the full configuration shape and
examples, use the [Configuration Files](configuration.md) guide.

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
[Protocol Contracts](protocols.md).
