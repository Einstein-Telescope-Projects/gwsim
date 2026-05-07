# Extensibility

`gwmock` can load third-party backends without any repository changes as long as
the backend satisfies the public protocol for its section.

## Resolution order

For `orchestration.*.backend`, gwmock resolves backends in this order:

1. Built-in aliases shipped by `gwmock`.
2. Python entry points in the `gwmock.population`, `gwmock.signal`, or
   `gwmock.noise` groups.
3. Direct `module:Class` references.
4. Legacy `module.Class` references, which still work but should be considered
   deprecated.

## Packaging a backend

Expose your class through an entry point:

```toml
[project.entry-points."gwmock.population"]
galactic_binaries = "my_pkg.populations:GalacticBinaryPopulation"
```

Then reference the alias from YAML:

```yaml
orchestration:
    population:
        backend: galactic_binaries
        source-type: bbh
        n-samples: 128
        arguments:
            catalog_path: /data/catalogs/galactic-binaries.h5
```

The same pattern works for signal and noise backends.

## Conformance checks

`gwmock` validates the resolved backend against the public contract for the
section it is loaded into:

- population backends must satisfy `GWPopSimulator`
- signal backends must provide the public `simulate(...)` surface
- noise backends must satisfy `NoiseSimulator`

If a class is missing a required member, the configuration should fail clearly
before any data is generated.

For the formal protocol definitions, see [Protocol Contracts](protocols.md). For
the adapter-backed configuration surface, see [Orchestration](orchestration.md).
