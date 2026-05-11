# Metadata Files

Provenance records are written alongside each simulated dataset and use the
`.metadata.json` suffix by default. The file format is determined by the suffix:
files ending in `.json` are written as JSON; files ending in any other suffix
(such as `.yaml`) are written as YAML. Both formats can be read back by
`gwmock simulate` for reproduction.

The `gwmock merge` command writes merged metadata with a `.metadata.yaml`
suffix.

See [Reproducibility](reproducibility.md) for the full schema, reproduction
recipe, and versioning rules.
