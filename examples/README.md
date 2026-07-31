# Example configurations

Ready-to-run configuration files for gwmock. Every directory here containing a
`config.yaml` is a **label** you can fetch from the CLI — the label is just the
path:

```bash
gwmock config --list                                    # show every label
gwmock config --get signal/bbh/et_triangle_sardinia --output config.yaml
gwmock simulate config.yaml
```

New to gwmock? Start with **`noise/uncorrelated_gaussian/quick_start`** — one
segment, signal and noise together, a few minutes to run. To write a
configuration from scratch instead, start from **`default_config`**, a commented
noise-only template.

These examples are sized to be _realistic_, not fast: most generate one day of
data in 4096-second segments. Shorten `total-duration` before running one end to
end.

## Choosing an example

**By what you want to generate**

| Goal                                      | Label                                          |
| ----------------------------------------- | ---------------------------------------------- |
| Detector noise only                       | `noise/uncorrelated_gaussian/<network>`        |
| Noise with transient glitches             | `noise/glitches/gengli/<network>/<detector>`   |
| CBC signals only (no noise)               | `signal/bbh/<network>`, `signal/bns/<network>` |
| Signals **and** noise together            | `noise/uncorrelated_gaussian/quick_start`      |
| Stochastic background                     | `signal/sgwb/<network>`                        |
| Signals from a different waveform library | `signal/waveform_backend/ripple`               |
| A blank starting point                    | `default_config`                               |

**By waveform library** — `signal.waveform-backend` chooses which library
generates the polarizations: `lal` (the default), `pycbc`, `ripple`, or
`gwsignal`. Any signal example accepts it. It names a **library, not a compute
device** — `ripple` generates ripple waveforms through the same per-event path,
since gwmock does not yet drive ripple's batched on-device path from a
configuration file.

**By detector network** — every `<network>` above is one of
`et_triangle_sardinia`, `et_triangle_emr`, `et_2l_aligned`, `et_2l_misaligned`.
These differ **only** in the `detectors:` entry, so you can equally take any
example and change that one field. The four are provided as separate files so
each is runnable without editing.

## Every example

| Label                                              | Generates          | Network                       | Notes                                                              |
| -------------------------------------------------- | ------------------ | ----------------------------- | ------------------------------------------------------------------ |
| `default_config`                                   | noise              | Triangle Sardinia             | Commented template; single 4096 s segment                          |
| `noise/uncorrelated_gaussian/quick_start`          | signal + noise     | Triangle Sardinia             | **Start here.** 1024 s, one BBH event                              |
| `noise/uncorrelated_gaussian/et_triangle_sardinia` | noise              | Triangle Sardinia             | 1 day, `ET_10_full_cryo_psd`                                       |
| `noise/uncorrelated_gaussian/et_triangle_emr`      | noise              | Triangle EMR                  | 1 day, `ET_10_full_cryo_psd`                                       |
| `noise/uncorrelated_gaussian/et_2l_aligned`        | noise              | 2L aligned                    | 1 day, `ET_15_full_cryo_psd`                                       |
| `noise/uncorrelated_gaussian/et_2l_misaligned`     | noise              | 2L misaligned                 | 1 day, `ET_15_full_cryo_psd`                                       |
| `noise/glitches/gengli/<network>/<e1\|e2\|e3>`     | noise + glitches   | all four                      | One file **per detector**; blip glitches at 1/min. Needs `gengli`  |
| `signal/bbh/<network>`                             | signal             | all four                      | `IMRPhenomXPHM`, f<sub>min</sub> 10 Hz, Earth rotation on          |
| `signal/bns/<network>`                             | signal             | all four                      | `IMRPhenomPv2_NRTidalv2`, f<sub>min</sub> 20 Hz, Earth rotation on |
| `signal/sgwb/<network>`                            | background + noise | Triangle Sardinia, 2L aligned | 16 s; signal written as **HDF5**, noise as GWF                     |
| `signal/waveform_backend/ripple`                   | signal             | Triangle Sardinia             | Waveforms from **ripple** rather than LAL. Needs `gwmock[jax]`     |

The glitch examples are per-detector rather than per-network because each
detector draws from its own glitch population file.

## The end-to-end test matrix

A **subset** of these examples is the end-to-end matrix: the configs driven
through the real CLI by the `e2e` test suite.

Those tests are excluded from the default run — they generate data — and run in
their own CI job with every extra installed. To run them yourself:

```bash
uv run pytest -m e2e --no-cov
```

The examples themselves are never edited. A test-time overlay
(`tests/e2e/overlay.py`) shortens each run and repoints its inputs at in-repo
files, so the examples stay realistic while the suite finishes in about two
minutes.

> **What is and is not checked.** Each entry is run and its output verified
> against the manifest the run itself records: every declared file present,
> every file decoding to finite samples, the right channels, and signal actually
> present where the span covers a population. Each entry is also run twice in
> separate processes and compared by content hash, so reproducibility is
> established.
>
> Not yet checked: agreement with **stored reference values**. That is the next
> step, and reproducibility was the prerequisite for it.
>
> The overlay shortens runs, which costs coverage worth naming: sampling
> frequency drops from 4096 Hz to 1024 Hz, moving the Nyquist frequency, and
> spans are cut, reducing segment counts. A defect appearing only at full rate
> or across many segments is out of scope here.

The subset is chosen to cover each distinct **code path** exactly once, not
every configuration. Where two examples differ only in values that flow through
the same code — four detector networks resolved by the same `Network` class, BBH
versus BNS both handled by `CBCSimulator` — one is enough, and unit tests are
what guarantee the others follow. That assumption is the reason the subset is
legitimate; if a change makes two examples take genuinely different paths, the
matrix needs a new entry.

| Label                                              | Code path it is intended to cover                                              |
| -------------------------------------------------- | ------------------------------------------------------------------------------ |
| `default_config`                                   | The blank template must run unedited; noise-only, single segment               |
| `noise/uncorrelated_gaussian/quick_start`          | Signal **and** noise in one run; CBC; GWF output                               |
| `noise/uncorrelated_gaussian/et_triangle_sardinia` | Noise-only across **many** segments (chunking, per-segment seeds)              |
| `signal/bbh/et_triangle_sardinia`                  | Signal-only CBC; Earth rotation; population loaded from file                   |
| `signal/sgwb/et_triangle_sardinia`                 | `StochasticBackgroundSimulator` — a different simulator class; **HDF5** output |
| `signal/waveform_backend/ripple`                   | A non-default waveform library resolved from config. Needs `ripplegw`          |
| `noise/glitches/gengli/et_triangle_sardinia/e1`    | **Not run** — glitch injection, blocked on `gengli` and a local glitch fixture |

Deliberately excluded, with the reason:

- **The other three networks**, for every category — same `Network` code,
  different numbers.
- **BNS**, because `resolve_simulator_backend` returns the same `CBCSimulator`
  for `bbh` and `bns`; they differ in waveform model and tidal parameters, which
  unit tests own. BNS is also the expensive case: its inspiral is roughly 160 s
  at 20 Hz, so it cannot be shortened to a test-sized segment without truncating
  the chirp and simulating something physically wrong.
- **`e2`/`e3` glitch files**, which differ only in the population file they
  read.

The tests will **not** edit these files. Durations, sample counts and input
paths are to be reduced by a test-time overlay, so the examples stay realistic
as examples while the suite stays fast. Changing an example will still change
what the suite runs — that is the point — but shortening one for the tests'
benefit should never be necessary.

Adding a new example is free: the label is derived from the directory path. Add
it to the table above. Add it to the matrix only if it reaches code no listed
entry does.
