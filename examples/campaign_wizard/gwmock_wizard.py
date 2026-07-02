#!/usr/bin/env python3
"""gwmock-wizard — deterministic interview -> gwmock campaign generator + launcher.

Author: Gianluca Inguglia <gianluca.inguglia@oeaw.ac.at>

No LLM anywhere in the run path. The flow is:

    interview   ask a few questions  -> writes a reproducible campaign spec (campaign.json)
    generate    campaign.json        -> per-chunk gwmock configs + a launcher script
    run         campaign.json        -> generate, then submit (SLURM) / execute (local pool)

Re-running `generate` on the same campaign.json is byte-identical: every chunk's
start-time and seed are derived deterministically from the spec, so the whole
campaign is reproducible and re-creatable from one small file.

Why chunking: each gwmock simulate run is independent. The wizard splits a
campaign into many independent `gwmock simulate <chunk>.yaml` invocations and
fans them out across a SLURM job array or a bounded local process pool, giving
parallelism across runs.

Stdlib only (json/argparse) so `generate` works even where gwmock is not
installed. gwmock itself is only needed at *run* time (the launcher calls
`gwmock simulate`), and it requires Python >= 3.12.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import textwrap

# --------------------------------------------------------------------------- #
# gwmock option space (values verified against gwmock v0.8.2 example configs)
# --------------------------------------------------------------------------- #
GEOMETRIES = {
    # name              -> gwmock network detector identifier (single string)
    "Triangle_EMR":      "ET-Triangle-EMR",
    "Triangle_Sardinia": "ET-Triangle-Sardinia",
    "2L_Aligned":        "ET-2L-Aligned",
    "2L_Misaligned":     "ET-2L-Misaligned",
}
PSDS = ["ET_10_full_cryo", "ET_15_full_cryo"]
GOALS = ["noise", "signal", "glitch", "mixture", "efficiency_far"]
WAVEFORMS = {  # source family (lowercase) -> (waveform_model, default population_file, default f_min)
    "bbh": ("IMRPhenomXPHM", "18321_1yrCatalogBBH.h5", 20),
    "bns": ("IMRPhenomPv2_NRTidalv2", "18321_1yrCatalogBNS.h5", 20),
}
GLITCH_POP_PLACEHOLDER = "<<SET_GLITCH_POPULATION_FILE>>"

DEFAULTS = {
    "sampling_frequency": 4096,
    "frame_duration": 4096,        # seconds per output frame
    "total_duration": "1 day",     # campaign length (signal/glitch/mixture)
    "start_time": 1577491218,      # base GPS
    "chunk_seconds": 4096,         # wall-clock length of one chunk (time-chunked goals)
    "n_seeds": 1,                  # independent realizations (noise / efficiency_far)
    "base_seed": 42,
    "source_family": "bbh",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parse_duration(v) -> int:
    """'1 day' / '6 hours' / '3600 s' / 3600 -> seconds (int)."""
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip().lower()
    if s.isdigit():
        return int(s)
    parts = s.split()
    if len(parts) == 2 and parts[0].replace(".", "", 1).isdigit():
        n = float(parts[0])
        unit = parts[1]
        # singularize plurals (hours->hour) without nuking the bare "s" abbreviation
        if len(unit) > 1 and unit.endswith("s"):
            unit = unit[:-1]
        mult = {"s": 1, "sec": 1, "second": 1, "min": 60, "minute": 60,
                "hour": 3600, "hr": 3600, "h": 3600, "day": 86400, "d": 86400,
                "week": 604800, "year": 31557600}.get(unit)
        if mult:
            return int(n * mult)
    raise ValueError(f"cannot parse duration {v!r} (try '1 day', '6 hours', '3600 s', or an int)")


def default_psd(geometry: str) -> str:
    """Geometry-based default sensitivity (both PSDs are confirmed gwmock names)."""
    if geometry in ("2L_Aligned", "2L_Misaligned"):
        return "ET_15_full_cryo"
    return "ET_10_full_cryo"


def build_glitches(spec):
    """One gengli glitch entry, emitted nested under noise.arguments.glitches."""
    return [{
        "kind": "gengli_blip",
        "rate": spec.get("glitch_rate", 0.016666667),
        "amplitude_distribution": {"distribution": "lognormal", "mean": 1.0, "std": 0.0},
        "population_file": spec.get("glitch_population_file", GLITCH_POP_PLACEHOLDER),
        "psd_file": f"{spec['psd']}_psd",
        "low_frequency_cutoff": 5.0,
    }]


# --------------------------------------------------------------------------- #
# YAML emitters (string-templated; we control the exact schema -> no yaml dep).
# Schema verified against gwmock v0.8.2 globals + orchestration example configs.
# --------------------------------------------------------------------------- #
def _globals_block(workdir, sf, frame_dur, start_time, total_dur=None):
    frame_eff = int(frame_dur)
    if total_dur is not None:
        # A final short time-chunk can leave total_dur < frame_dur; clamp the
        # per-frame duration so total-duration >= duration always holds.
        frame_eff = min(int(frame_dur), int(total_dur))
    lines = [
        "globals:",
        "  simulator-arguments:",
        f"    sampling-frequency: {sf}",
        f"    duration: {frame_eff}",
    ]
    if total_dur is not None:
        lines.append(f"    total-duration: {int(total_dur)}")
    lines += [
        f"    start-time: {start_time}",
        f'  working-directory: "{workdir}"',
        '  output-directory: "output"',
        '  metadata-directory: "metadata"',
    ]
    return "\n".join(lines)


def _det_list(detectors, indent):
    pad = " " * indent
    return "\n".join(f"{pad}- {d}" for d in detectors)


def _glitches_block(glitches, indent):
    """Render a glitch list as YAML nested under noise.arguments (keyword at `indent`)."""
    pad = " " * indent
    item = " " * (indent + 2)
    key = " " * (indent + 4)
    lines = [f"{pad}glitches:"]
    for g in glitches:
        ad = g["amplitude_distribution"]
        lines.append(f'{item}- kind: {g["kind"]}')
        lines.append(f'{key}rate: {g["rate"]}')
        lines.append(
            f'{key}amplitude_distribution: {{ distribution: {ad["distribution"]}, '
            f'mean: {ad["mean"]}, std: {ad["std"]} }}'
        )
        lines.append(f'{key}population_file: {g["population_file"]}')
        lines.append(f'{key}psd_file: {g["psd_file"]}')
        lines.append(f'{key}low_frequency_cutoff: {g["low_frequency_cutoff"]}')
    return "\n".join(lines)


def signal_config(spec, workdir, start_time, total_dur):
    fam = spec["source_family"]
    wf_model, _, fmin = WAVEFORMS[fam]
    pop = spec["population_file"]
    net = GEOMETRIES[spec["geometry"]]
    fam_upper = fam.upper()
    g = _globals_block(workdir, spec["sampling_frequency"], spec["frame_duration"],
                       start_time, total_dur)
    return f"""{g}

orchestration:
  population:
    backend: FilePopulationLoader
    source-type: {fam}
    n-samples: {spec.get('n_samples', 1)}
    arguments:
      path: {pop}
  signal:
    waveform-model: {wf_model}
    minimum-frequency: {spec.get('minimum_frequency', fmin)}
    earth-rotation: true
    detectors:
{_det_list([net], 6)}
    output:
      output_directory: signal
      file_name: 'E-{{{{ detectors }}}}_STRAIN_{fam_upper}-{{{{ start_time }}}}-{{{{ duration }}}}.gwf'
      arguments:
        channel: '{{{{ detectors }}}}:STRAIN'
"""


def noise_config(spec, workdir, start_time, seed, total_dur, glitches=None):
    net = GEOMETRIES[spec["geometry"]]
    psd = f"{spec['psd']}_psd"
    g = _globals_block(workdir, spec["sampling_frequency"], spec["frame_duration"],
                       start_time, total_dur)
    body = f"""{g}

orchestration:
  noise:
    arguments:
      psd_file: {psd}
      seed: {seed}
      minimum_frequency: {spec.get('minimum_frequency', 20)}
      detectors:
{_det_list([net], 8)}"""
    if glitches:
        body += "\n" + _glitches_block(glitches, 6)
    body += f"""
    output:
      output_directory: noise
      file_name: 'E-{{{{ detectors }}}}_STRAIN_NOISE-{{{{ start_time }}}}-{{{{ duration }}}}.gwf'
      arguments:
        channel: '{{{{ detectors }}}}:STRAIN'
"""
    return body


# --------------------------------------------------------------------------- #
# chunk planning
# --------------------------------------------------------------------------- #
def plan_time_chunks(spec):
    """Contiguous time partition: (index, start_time, chunk_seconds)."""
    total = parse_duration(spec["total_duration"])
    step = int(spec["chunk_seconds"])
    base = int(spec["start_time"])
    n = max(1, math.ceil(total / step))
    out = []
    for k in range(n):
        st = base + k * step
        dur = min(step, total - k * step)
        out.append((k, st, dur))
    return out


def plan_seed_chunks(spec):
    """Independent noise realizations: (index, start_time, seed, duration)."""
    duration = parse_duration(spec["total_duration"])
    base_seed = int(spec["base_seed"])
    base_t = int(spec["start_time"])
    return [(k, base_t, base_seed + k, duration) for k in range(int(spec["n_seeds"]))]


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #
def generate(spec, out_root):
    name = spec["name"]
    camp_dir = os.path.join(out_root, name)
    chunk_dir = os.path.join(camp_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    goal = spec["goal"]
    configs = []  # (relative_config_path, workdir)

    def emit(fname, text, workdir):
        path = os.path.join(chunk_dir, fname)
        with open(path, "w") as fh:
            fh.write(text)
        configs.append((path, workdir))

    if goal in ("signal", "glitch", "mixture"):
        for (k, st, dur) in plan_time_chunks(spec):
            wd_base = os.path.join(camp_dir, "out", f"chunk_{k:04d}")
            if goal in ("signal", "mixture"):
                emit(f"signal_{k:04d}.yaml",
                     signal_config(spec, wd_base + "_signal", st, dur), wd_base + "_signal")
            if goal in ("glitch", "mixture"):
                # gwmock has no standalone glitch run: glitches ride inside a
                # noise config, so a glitch goal also produces a noise floor.
                emit(f"noise_{k:04d}.yaml",
                     noise_config(spec, wd_base + "_noise", st,
                                  int(spec["base_seed"]) + 10_000 + k, dur,
                                  glitches=build_glitches(spec)), wd_base + "_noise")

    elif goal == "noise":
        for (k, st, seed, dur) in plan_seed_chunks(spec):
            wd = os.path.join(camp_dir, "out", f"seed_{k:04d}")
            emit(f"noise_{k:04d}.yaml", noise_config(spec, wd, st, seed, dur), wd)

    elif goal == "efficiency_far":
        # background: noise seed-sweep
        for (k, st, seed, dur) in plan_seed_chunks(spec):
            wd = os.path.join(camp_dir, "out", f"bg_seed_{k:04d}")
            emit(f"bg_noise_{k:04d}.yaml", noise_config(spec, wd, st, seed, dur), wd)
        # foreground: time-chunked signal injections
        for (k, st, dur) in plan_time_chunks(spec):
            wd = os.path.join(camp_dir, "out", f"fg_chunk_{k:04d}_signal")
            emit(f"fg_signal_{k:04d}.yaml", signal_config(spec, wd, st, dur), wd)
    else:
        raise ValueError(f"unknown goal {goal!r}")

    # config manifest (one path per line) -> drives both launchers
    manifest = os.path.join(camp_dir, "configs.txt")
    with open(manifest, "w") as fh:
        for path, _ in configs:
            fh.write(os.path.abspath(path) + "\n")

    # launchers
    write_local_launcher(camp_dir, manifest, spec)
    write_slurm_launcher(camp_dir, manifest, spec, len(configs))
    if goal == "mixture":
        write_merge_note(camp_dir, configs)

    # freeze the resolved spec next to the outputs for provenance
    with open(os.path.join(camp_dir, "campaign.resolved.json"), "w") as fh:
        json.dump(spec, fh, indent=2)

    return camp_dir, len(configs)


def write_local_launcher(camp_dir, manifest, spec):
    extra = spec.get("simulate_extra_args", "")
    nproc = spec.get("local_workers", "$(nproc)")
    sh = f"""#!/usr/bin/env bash
# Local bounded-parallel runner. Each line of configs.txt is an independent
# `gwmock simulate`. Re-run to resume (gwmock checkpoints completed frames).
set -euo pipefail
MANIFEST="{os.path.abspath(manifest)}"
WORKERS="${{GWMOCK_WORKERS:-{nproc}}}"
echo "[gwmock-wizard] $(wc -l < "$MANIFEST") chunks, ${{WORKERS}} workers"
xargs -a "$MANIFEST" -P "$WORKERS" -I {{}} bash -c \\
  'echo "[start] {{}}"; gwmock simulate "{{}}" {extra} && echo "[done] {{}}"'
echo "[gwmock-wizard] all chunks finished"
"""
    p = os.path.join(camp_dir, "run_local.sh")
    with open(p, "w") as fh:
        fh.write(sh)
    os.chmod(p, 0o755)


def write_slurm_launcher(camp_dir, manifest, spec, n):
    sl = spec.get("slurm", {})
    extra = spec.get("simulate_extra_args", "")
    directives = "\n".join([
        f"#SBATCH --job-name=gwmock_{spec['name']}",
        f"#SBATCH --array=0-{n-1}%{sl.get('max_concurrent', n)}",
        f"#SBATCH --partition={sl.get('partition', '<<PARTITION>>')}",
        f"#SBATCH --account={sl.get('account', '<<ACCOUNT>>')}",
        f"#SBATCH --time={sl.get('time', '04:00:00')}",
        f"#SBATCH --mem={sl.get('mem', '16G')}",
        f"#SBATCH --cpus-per-task={sl.get('cpus', 4)}",
        f"#SBATCH --output={os.path.abspath(camp_dir)}/logs/chunk_%a.out",
    ])
    sh = f"""#!/usr/bin/env bash
{directives}
# One array task per chunk. Resubmit to resume (gwmock checkpoints completed frames).
set -euo pipefail
mkdir -p "{os.path.abspath(camp_dir)}/logs"
MANIFEST="{os.path.abspath(manifest)}"
CFG=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MANIFEST")
echo "[task ${{SLURM_ARRAY_TASK_ID}}] gwmock simulate ${{CFG}}"
gwmock simulate "${{CFG}}" {extra}
"""
    p = os.path.join(camp_dir, "submit_slurm.sbatch")
    with open(p, "w") as fh:
        fh.write(sh)


def write_merge_note(camp_dir, configs):
    note = textwrap.dedent("""\
        # mixture goal: for each time-chunk the wizard emits a signal frame and a
        # separate noise frame (the noise carries the gengli glitches). gwmock does
        # not superpose them in one run, so combine the matching per-detector frames
        # after all chunks finish.
        #
        # `gwmock merge` exists as a subcommand, but its exact arguments are not
        # exercised here. Check them against your gwmock version before relying on it:
        #     gwmock merge --help
        # then superpose the matching {signal, noise} frames per chunk/detector.
        """)
    with open(os.path.join(camp_dir, "MERGE_README.txt"), "w") as fh:
        fh.write(note)


# --------------------------------------------------------------------------- #
# interview
# --------------------------------------------------------------------------- #
def _ask(prompt, default=None, choices=None):
    suffix = f" [{default}]" if default is not None else ""
    if choices:
        print(f"  options: {', '.join(map(str, choices))}")
    while True:
        ans = input(f"{prompt}{suffix}: ").strip()
        if not ans and default is not None:
            return default
        if choices and ans not in choices:
            print(f"  -> pick one of: {', '.join(map(str, choices))}")
            continue
        if ans:
            return ans


def interview():
    print("=== gwmock-wizard interview ===  (Enter accepts the [default])\n")
    spec = {}
    spec["name"] = _ask("Campaign name", "my_et_campaign")
    spec["goal"] = _ask("Goal", "signal", GOALS)
    spec["geometry"] = _ask("ET geometry", "Triangle_EMR", list(GEOMETRIES))
    spec["psd"] = _ask("Sensitivity (PSD)", default_psd(spec["geometry"]), PSDS)
    if spec["goal"] in ("signal", "mixture", "efficiency_far"):
        spec["source_family"] = _ask("Source family", "bbh", list(WAVEFORMS))
        _, defpop, _ = WAVEFORMS[spec["source_family"]]
        spec["population_file"] = _ask("Population file (HDF5 path or URL)", defpop)
    if spec["goal"] in ("glitch", "mixture"):
        spec["glitch_population_file"] = _ask(
            "Glitch population file (HDF5 path or URL)", GLITCH_POP_PLACEHOLDER)
    spec["sampling_frequency"] = int(_ask("Sampling frequency [Hz]", DEFAULTS["sampling_frequency"]))
    spec["frame_duration"] = int(_ask("Frame duration [s]", DEFAULTS["frame_duration"]))
    if spec["goal"] in ("signal", "glitch", "mixture", "efficiency_far"):
        spec["total_duration"] = _ask("Total duration", DEFAULTS["total_duration"])
        spec["chunk_seconds"] = int(_ask("Seconds per chunk", DEFAULTS["chunk_seconds"]))
    if spec["goal"] in ("noise", "efficiency_far"):
        spec["n_seeds"] = int(_ask("Number of noise realizations (seeds)", DEFAULTS["n_seeds"]))
        if "total_duration" not in spec:
            spec["total_duration"] = _ask("Duration per realization", DEFAULTS["total_duration"])
    spec["start_time"] = int(_ask("Base GPS start-time", DEFAULTS["start_time"]))
    spec["base_seed"] = int(_ask("Base seed", DEFAULTS["base_seed"]))
    spec["backend"] = _ask("Compute backend", "local", ["local", "slurm"])
    if spec["backend"] == "slurm":
        spec["slurm"] = {
            "partition": _ask("  SLURM partition", "<<PARTITION>>"),
            "account": _ask("  SLURM account", "<<ACCOUNT>>"),
            "time": _ask("  Walltime per chunk", "04:00:00"),
            "mem": _ask("  Memory per chunk", "16G"),
            "cpus": int(_ask("  CPUs per chunk", 4)),
        }
    # fill remaining defaults
    for k, v in DEFAULTS.items():
        spec.setdefault(k, v)

    out = f"{spec['name']}.campaign.json"
    with open(out, "w") as fh:
        json.dump(spec, fh, indent=2)
    print(f"\nWrote {out}")
    print(f"  generate:  {sys.argv[0]} generate {out}")
    print(f"  run:       {sys.argv[0]} run {out}")
    return out, spec


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def load_spec(path):
    with open(path) as fh:
        return json.load(fh)


def cmd_generate(args):
    spec = load_spec(args.campaign)
    camp_dir, n = generate(spec, args.out_root)
    print(f"Generated {n} chunk config(s) under {camp_dir}/chunks/")
    print(f"  manifest:        {camp_dir}/configs.txt")
    print(f"  local launcher:  {camp_dir}/run_local.sh   (GWMOCK_WORKERS=N to cap parallelism)")
    print(f"  slurm launcher:  {camp_dir}/submit_slurm.sbatch")
    return camp_dir, spec


def cmd_run(args):
    camp_dir, spec = cmd_generate(args)
    backend = spec.get("backend", "local")
    if backend == "slurm":
        cmd = f"sbatch {os.path.join(camp_dir, 'submit_slurm.sbatch')}"
    else:
        cmd = f"bash {os.path.join(camp_dir, 'run_local.sh')}"
    if not args.submit:
        print(f"\n[dry-run] to launch ({backend}):\n    {cmd}\n  (add --submit to launch now)")
        return
    print(f"\n[submit] {cmd}")
    os.system(cmd)


def main():
    ap = argparse.ArgumentParser(description="gwmock-wizard: interview -> gwmock campaign -> launcher")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("interview", help="ask questions and write a campaign spec")
    sub.add_parser("show-options", help="print the available geometries / PSDs / goals")

    g = sub.add_parser("generate", help="campaign.json -> chunk configs + launchers")
    g.add_argument("campaign")
    g.add_argument("--out-root", default="./campaigns")

    r = sub.add_parser("run", help="generate, then submit (SLURM) / execute (local)")
    r.add_argument("campaign")
    r.add_argument("--out-root", default="./campaigns")
    r.add_argument("--submit", action="store_true", help="actually launch (otherwise dry-run)")

    args = ap.parse_args()
    if args.cmd == "interview":
        interview()
    elif args.cmd == "show-options":
        print("goals:     ", ", ".join(GOALS))
        print("geometries:", ", ".join(GEOMETRIES))
        print("psds:      ", ", ".join(PSDS))
        print("sources:   ", ", ".join(WAVEFORMS))
    elif args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
