# Publishing Simulation Data to Zenodo

The `gwmock repository` command suite allows you to create, manage, and publish
your gravitational-wave simulation datasets to [Zenodo](https://zenodo.org), a
community-driven open-access repository. This enables long-term preservation,
DOI assignment, and easy sharing with the GW community.

## Overview

Publishing a dataset involves these steps:

1. **Create** a new deposition (draft repository)
2. **Upload** your simulation files (GWF frames, metadata, configs)
3. **Update** metadata (title, authors, keywords, etc.)
4. **Publish** to finalize (generates a persistent DOI)
5. **Download** published records (by anyone, no token needed)

## Setup

### Get an API Token

Before using the repository commands, you need a Zenodo API token:

1. **Production Zenodo:** Go to
   [https://zenodo.org/account/settings/applications/tokens/new](https://zenodo.org/account/settings/applications/tokens/new)
2. **Sandbox (Testing):** Go to
   [https://sandbox.zenodo.org/account/settings/applications/tokens/new](https://sandbox.zenodo.org/account/settings/applications/tokens/new)

When creating a token, ensure it has these scopes:

- `deposit:write` — Write access to create/upload files
- `deposit:actions` — Permission to publish depositions

### Set Environment Variables

Store your tokens securely as environment variables:

```bash
# Production token
export ZENODO_API_TOKEN="your_production_token_here"

# Sandbox token (for testing)
export ZENODO_SANDBOX_API_TOKEN="your_sandbox_token_here"
```

**Tip**: Add these to your .bashrc, .zshrc, or .env file to avoid re-entering
them.

## Verify Your Token

Test that your token works before publishing:

```bash
# Verify production token
gwmock repository verify

# Verify sandbox token
gwmock repository verify --sandbox
```

If successful, you'll see:

```text
✓ Token is valid!
  Environment: Zenodo (Production)
  Found 3 draft deposition(s)
```

## Workflow

Step 1: Create a Deposition

Start by creating a new draft deposition:

```bash
gwmock repository create \
  --title "GW Mock Data Challenge v1" \
  --description "Simulated binary black hole coalescences for ET"
```

**Interactive mode**: Omit options to be prompted:

```bash
gwmock repository create
# Deposition Title: GW Mock Data Challenge v1
# Deposition Description: Simulated binary black hole coalescences
```

**Output**:

```text
Creating deposition...
✓ Deposition created successfully!
  ID: 123456
  Next: gwmock repository upload 123456 --file <path>
```

Save the deposition ID (e.g., `123456`) for subsequent commands.

Step 2: Upload Files

Upload your simulation outputs and metadata:

```bash
# Single file
gwmock repository upload 123456 --file simulation_output.hdf5

# Multiple files
gwmock repository upload 123456 \
  --file simulation_output.hdf5 \
  --file metadata.yaml \
  --file config.yaml
```

**Features**:

- Files are uploaded with automatic timeout adjustment (10 seconds per MB)
- Progress bar shows upload status
- Failed uploads are reported; retry-safe via exponential backoff

**Output**:

```shell
Uploading 3 file(s) to deposition 123456...
Uploading ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
✓ simulation_output.hdf5 (245.50 MB)
✓ metadata.yaml (0.05 MB)
✓ config.yaml (0.02 MB)
Next: gwmock repository update <id> --metadata-file <file>
```

Step 3: Update Metadata

Enhance your deposition with structured metadata:

```yaml
creators:
    - name: 'Jane Doe'
      affiliation: 'LIGO Laboratory'
      orcid: '0000-0000-0000-0000'
    - name: 'John Smith'
      affiliation: 'Virgo Collaboration'

keywords:
    - 'gravitational waves'
    - 'mock data challenge'
    - 'binary black holes'
    - 'LIGO'
    - 'Virgo'

license: 'cc-by-4.0'
contributors:
    - name: 'LIGO Laboratory'
      role: 'Hosting institution'

related_identifiers:
    - identifier: '10.7935/gqm7-wf12'
      relation: 'references'
      resource_type: 'publication'
```

**Upload metadata**:

```bash
gwmock repository update 123456 --metadata-file deposition_metadata.yaml
```

**Output**:

```shell
Updating metadata for deposition 123456...
✓ Metadata updated successfully
Next: gwmock repository publish 123456
```

Step 4: Publish

Once files and metadata are complete, publish your deposition:

```bash
gwmock repository publish 123456
```

**Confirmation prompt**:

```shell
Publish deposition 123456? This action is permanent and cannot be undone. [y/N]:
```

**Output (on success)**:

```shell
Publishing deposition 123456...
✓ Published successfully!
  DOI: 10.5281/zenodo.123456
```

**Important**: Publishing is permanent. Once published, you cannot modify files
or delete the record. Always verify metadata before publishing.

Step 5: Share & Download

Your dataset now has a permanent DOI and is discoverable:

Share the DOI: `10.5281/zenodo.123456`

Download files (anyone can do this without a token):

```bash
gwmock repository download 123456 --file simulation_output.hdf5 --output ./data.hdf5
```

## Advanced Usage

Testing with Sandbox

Use the Zenodo Sandbox to test your workflow before publishing to production:

```bash
# Create in sandbox
gwmock repository create \
  --title "Test Dataset" \
  --sandbox

# Upload files
gwmock repository upload 123456 --file data.hdf5 --sandbox

# Publish to sandbox
gwmock repository publish 123456 --sandbox
```

**Sandbox DOI example**: `10.5072/zenodo.123456` (note the `10.5072/` prefix)

List Your Depositions

View all your depositions:

```bash
# List published records
gwmock repository list

# List draft (unpublished) records
gwmock repository list --status draft

# List in sandbox
gwmock repository list --status draft --sandbox
```

Output:

```shell
Listing published depositions...

┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ ID         ┃ Title                              ┃ DOI                   ┃ Created    ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 123456     │ GW Mock Data Challenge v1          │ 10.5281/zenodo.123456 │ 2024-01-15 │
│ 123455     │ BBH Parameter Study                │ 10.5281/zenodo.123455 │ 2024-01-10 │
└────────────┴────────────────────────────────────┴───────────────────────┴────────────┘
```

Delete a Draft

Remove an unpublished deposition:

```bash
gwmock repository delete 123456

# Skip confirmation
gwmock repository delete 123456 --force
```

**Note**: Only unpublished (draft) depositions can be deleted. Published records
are permanent.

Download Existing Records

Download files from any published record using the deposition ID:

```bash
# Download a file
gwmock repository download 123456 \
  --file simulation_output.hdf5 \
  --output ./downloaded_data.hdf5

# Specify file size for faster timeout tuning
gwmock repository download 123456 \
  --file large_dataset.hdf5 \
  --output ./large_dataset.hdf5 \
  --file-size-mb 5000
```

## Metadata Best Practices

When publishing GW simulation data, include:

1. Title: Clear, descriptive (e.g., "GW Mock Data Challenge v1: Binary Black
   Holes")
2. Description: Simulation parameters, instruments, frequency range
3. Creators: Full names and ORCiDs (if available)
4. Keywords: gravitational waves, detector names (LIGO, Virgo), signal types
   (binary black holes, neutron stars)
5. License: Recommend cc-by-4.0 for open science
6. Related Identifiers: Link to papers, talks, or other related datasets

**Example**:

```yaml
title: 'GW Mock Data Challenge v1: Synthetic Binary Black Hole Signals'

description: |
    Simulated gravitational-wave strain data for LIGO Hanford, LIGO Livingston,
    and Virgo detectors. Includes 1000 binary black hole coalescence waveforms
    with varying masses (10-100 solar masses), spins, and sky positions.

    Sampling rate: 16384 Hz
    Duration: 8 seconds per event
    Frequency range: 20-512 Hz

    Generated using PyCBC v1.18.4 and LALSuite v7.0.

creators:
    - name: 'Jane Doe'
      orcid: '0000-0001-2345-6789'
      affiliation: 'LIGO Laboratory, Caltech'

keywords:
    - 'gravitational waves'
    - 'LIGO'
    - 'Virgo'
    - 'binary black holes'
    - 'mock data challenge'
    - 'synthetic data'

license: 'cc-by-4.0'
```

## Troubleshooting

### 403 Forbidden Error

**Problem**: Publishing or listing fails with 403 Client Error: FORBIDDEN

**Solutions**:

1. Verify your token is valid: `gwmock repository verify`
2. Generate a new token from
   [https://zenodo.org/account/settings/applications/tokens/new](https://zenodo.org/account/settings/applications/tokens/new)
3. Ensure the token has `deposit:write` and `deposit:actions` scopes
4. Check that you're using the correct environment (--sandbox for sandbox, omit
   for production)

### Token Not Found

**Problem: Error**: No Zenodo access token provided

**Solution**: Set environment variables:

```bash
export ZENODO_API_TOKEN="your_token"
export ZENODO_SANDBOX_API_TOKEN="your_sandbox_token"
```

### Upload Timeout

**Problem**: Large files fail with timeout errors

**Solution**: The CLI auto-adjusts timeouts based on file size (10 seconds per
MB). For very large files (> 10 GB), you can manually specify:

```bash
gwmock repository upload 123456 --file huge_file.hdf5
```

The retry logic with exponential backoff will automatically retry on transient
failures.

### Cannot Modify After Publishing

**Problem**: Need to fix metadata after publishing

**Solution**: Create a new deposition. Zenodo treats each published version as
immutable. You can:

1. Create a new deposition with updated metadata
2. Link it to the previous version using `related_identifiers` in metadata
3. Publish with an increment (e.g., "v2")
