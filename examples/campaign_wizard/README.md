# gwmock campaign wizard

`gwmock_wizard.py` is a small deterministic helper that turns a short
interview into a reproducible gwmock simulation campaign. It splits the
campaign into independent chunks so the work can run in parallel, and it
writes ready-to-run launchers for either a bounded local process pool or a
SLURM job array. There is no language model anywhere in the run path; the
wizard only asks a few questions and writes plain files.

The workflow has three steps. First you run `interview`, which asks a handful
of questions and saves the answers to `<name>.campaign.json`. Then you run
`generate`, which turns that campaign file into per-chunk gwmock YAML configs
plus the launcher scripts. Finally you `run`, which does the generate step and
then launches the chunks, locally or on SLURM. Running `generate` again on the
same campaign file produces byte-identical output, because every chunk's start
time and random seed are derived from the spec rather than from the clock.

## Requirements

Python 3.12 or newer. The `interview` and `generate` steps use only the
standard library, so you can design and inspect a campaign on a machine that
does not have gwmock installed. gwmock itself is only needed at run time,
because the launchers call `gwmock simulate` on each generated config.

## Install

The recommended path is to install from PyPI with `uv`. The bundled script

```bash
./install_gwmock.sh
```

creates `./gwmock-env` with Python 3.13, installs gwmock into it, and verifies
the install. Activate it with `source gwmock-env/bin/activate`. To work from a
source checkout instead, run `./install_gwmock.sh --source`, which clones this
repository and installs the clone into the environment. See the top-level
gwmock README for the manual installation recipe.

## Quickstart

A good first run is a small noise-only campaign. Activate the environment,
then start the interview:

```bash
source gwmock-env/bin/activate
python3 gwmock_wizard.py interview
```

Accept the defaults, choose the goal `noise`, the geometry `Triangle_EMR`, and
the PSD `ET_10_full_cryo`, and give it a short duration such as 256 seconds.
That writes a campaign file. Generate and launch it locally with

```bash
python3 gwmock_wizard.py run <name>.campaign.json --submit
```

When the chunks finish, look under `campaigns/<name>/out/` for the resulting
`.gwf` frames. A ready-made spec is provided in `example.campaign.json`.

## Commands

```bash
python3 gwmock_wizard.py interview                      # write a campaign spec
python3 gwmock_wizard.py show-options                   # list goals, geometries, PSDs
python3 gwmock_wizard.py generate <spec>.campaign.json  # chunk configs + launchers
python3 gwmock_wizard.py run <spec>.campaign.json       # generate, then launch
```

`run` is a dry run by default; add `--submit` to actually launch. The
`--out-root` option controls where campaigns are written and defaults to
`./campaigns`.

## Output layout

Each campaign lands in `campaigns/<name>/`. Inside it, `chunks/` holds the
per-chunk gwmock YAML configs, and `configs.txt` is a manifest with one config
path per line that drives both launchers. `run_local.sh` runs the chunks
through a local process pool, where `GWMOCK_WORKERS=N` caps how many run at
once. `submit_slurm.sbatch` is a SLURM array job; fill in the partition and
account placeholders before submitting. `campaign.resolved.json` records the
fully resolved spec for provenance, and `out/` collects the simulation frames.

## Goals

- `noise` produces colored detector noise from a PSD and needs no external
  catalog.
- `signal` produces CBC injections and needs a population file, which is
  passed through as the population path.
- `glitch` produces gengli glitches; since glitches are configured inside the
  noise orchestration block, a glitch run also produces a noise floor.
- `mixture` emits a signal frame and a noise-with-glitches frame separately
  for each chunk, to be combined afterward; see the generated
  `MERGE_README.txt`.
- `efficiency_far` produces a background noise seed sweep together with signal
  injections.

## Geometries and PSDs

The wizard knows four Einstein Telescope network geometries, each mapped to a
single gwmock network identifier: `Triangle_EMR` maps to `ET-Triangle-EMR`,
`Triangle_Sardinia` to `ET-Triangle-Sardinia`, `2L_Aligned` to
`ET-2L-Aligned`, and `2L_Misaligned` to `ET-2L-Misaligned`.

Two sensitivity curves are offered, `ET_10_full_cryo` and `ET_15_full_cryo`.
The interview defaults the triangle geometries to the former and the 2L
geometries to the latter. In the generated config the PSD appears as a bare
name with a `_psd` suffix, for example `ET_10_full_cryo_psd`.

Durations may be written as `1 day`, `6 hours`, `3600 s`, or a bare integer
number of seconds. The wizard always emits bare integer seconds into the
gwmock config.

## Reproducibility

The campaign spec `<name>.campaign.json` is the single source of truth.
Commit it, and anyone can recreate the identical campaign, because the chunk
start times and seeds are derived deterministically from the spec.
