# Validating Output Files

The `gwmock validate` command verifies the integrity of generated simulation
files by comparing their SHA-256 hashes against the expected values stored in
the corresponding `.metadata.json` files.

Use it after a simulation completes, after copying files, or before publishing a
dataset to confirm no files were corrupted or accidentally overwritten.

## Basic Usage

```bash
gwmock validate [OPTIONS] PATHS...
```

`PATHS` can be any mix of:

- Individual output files (`.hdf5`, etc.)
- Individual metadata files (`.metadata.json` or `.metadata.yaml`)
- Directories — scanned automatically for both file types

### Validate a whole simulation run

Pass the output and metadata directories together:

```bash
gwmock validate output/ metadata/
```

### Validate specific output files

gwmock finds the corresponding metadata files automatically:

```bash
gwmock validate output/noise/E-ET1_SARD_STRAIN_NOISE-1577491218-1024.hdf5 output/signal/E-ET1_SARD_STRAIN_BBH-1577491218-1024.hdf5
```

### Validate from metadata files directly

```bash
gwmock validate metadata/orchestration-0.metadata.json
```

### Validate a subset by pattern

Use `--pattern` to match only noise files, for example:

```bash
gwmock validate output/ --pattern "*noise*"
```

## Options

| Option                    | Description                                                   |
| ------------------------- | ------------------------------------------------------------- |
| `--metadata-paths TEXT`   | Additional metadata files or directories to load hashes from  |
| `--pattern TEXT`          | Glob pattern to match output files (e.g. `*noise*`)           |
| `--metadata-pattern TEXT` | Glob pattern to match metadata files (default: `*metadata.*`) |

## Exit Codes

`gwmock validate` exits with code `0` if all files pass and code `1` if any hash
mismatches are detected or files are missing.
