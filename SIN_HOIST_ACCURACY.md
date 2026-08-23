# Which float64 `sin(pi*x)` is more accurate: direct, or hoisted out of the tap loop?

A windowed-sinc resample tap loop evaluates `sin(pi*x)` once per tap, at
`x = k + frac` — `k` the integer tap offset, `frac` the fractional sample phase.
Since

```text
sin(pi*(k + frac)) = (-1)**k * sin(pi*frac)
```

is an identity in real arithmetic, the transcendental can be hoisted: evaluate
the sine once per loop and turn it into each tap's value with a sign flip, which
is exact in IEEE 754. That replaces 127 sine calls with one. It is not
bit-identical to the direct form, which raised the question this study answers:
**which form is closer to the true value?**

Everything below is measured against an mpmath reference at 50 decimal digits,
over 45 837 inputs. The script is `scripts/sin_hoist_accuracy.py`; the command
to rerun it is at the end.

## Bottom line

**The hoisted form is more accurate, decisively and systematically — not
indistinguishable.** Over the pooled sweep, taking only the 31 578 inputs where
an ulp count is meaningful (see [Metrics](#metrics)):

| pooled metric                    | `direct` = `sin(pi*x)` | `hoist[0,1)` = `(-1)**floor(x) * sin(pi*frac)` |
| -------------------------------- | ---------------------- | ---------------------------------------------- |
| median error                     | **71.2 ulps**          | **0.400 ulps**                                 |
| mean error                       | 3.32e6 ulps            | 11.3 ulps                                      |
| max error                        | 1.02e10 ulps           | 1.31e3 ulps                                    |
| max absolute error               | 2.9e-9                 | 3.43e-16                                       |
| strictly closer to the reference | 634 inputs (1.38%)     | 43 815 inputs (95.59%)                         |

In the regime the kernel actually occupies — a 127-tap loop, so `|k| <= 63` —
the hoisted form's median error is **0.399 ulps against 28.8 ulps**, it is
closer at 94.54% of inputs against 1.81%, and where it wins it wins by a median
factor of **62x** (median margin 32.8 ulps, worst 9.29e4 ulps). Where the direct
form wins it wins by a median factor of 2.4x (median margin 1 ulp).

The reason is structural, not statistical: `float64(pi)` falls short of pi by
1.2246e-16, so `sin(fl(pi)*x)` evaluates the sine at an argument displaced by
about `|x| * 1.2246e-16`. That displacement grows with `|x|`; the hoisted form's
argument never leaves `[0, 1)`, so its displacement does not grow at all. **The
hoist's advantage is therefore a factor of roughly `|x|`**, confirmed directly
below.

**A third form is better still.** Reducing to `[-1/2, 1/2]` instead of `[0, 1)`
— i.e. `n = round(x)`, `r = x - n`, value `(-1)**n * sin(pi*r)` — costs exactly
the same one transcendental per loop and the same per-tap sign flip, and it is
**never worse than 1.53 ulps at any of the 45 837 inputs** — including the ones
adjacent to a zero, where the other two forms reach 1e15 ulps — with a max
absolute error of 1.24e-16. It removes the one weakness the `[0, 1)` hoist has:
at `frac -> 1` the reduced argument sits next to pi, where rounding
`fl(pi)*frac` costs the result its leading digits. If the hoist is adopted, this
is the variant to adopt.

## What was compared

| form              | expression                                             | sines per 127-tap loop |
| ----------------- | ------------------------------------------------------ | ---------------------- |
| `direct`          | `sin(pi*x)`                                            | 127                    |
| `hoist[0,1)`      | `(-1)**floor(x) * sin(pi*frac)`, `frac = x - floor(x)` | 1                      |
| `hoist[-1/2,1/2]` | `(-1)**n * sin(pi*r)`, `n = round(x)`, `r = x - n`     | 1                      |

`r` is as loop-invariant as `frac` is (`round` splits the phase once, before the
loop), and the per-tap sign is still a parity flip of the tap index, so the
third form saves the same 126 of 127 transcendentals as the second. Its extra
cost over the second is one `round` instead of one `floor` per loop.

## How it was measured

- **Reference**: mpmath at 50 decimal digits (~166 bits), `pi` carried at the
  same precision, evaluated from the exact reduced argument so that the
  reference is _exactly_ zero at integer `x`.
- **Reference validated, not assumed**: the reduced reference agrees with an
  unreduced high-precision `sin(pi*x)` computed at 120 digits to a maximum
  **relative discrepancy of 2.6e-51** across every group. That is 35 orders of
  magnitude below the ~1e-16 effects under study.
- **The reductions were verified exact, not argued**: `x - floor(x)`,
  `x - round(x)` and `frac - round(frac)` were each checked in high precision at
  every input — **137 511 of 137 511 exact, zero violations**. This matters for
  fairness: if a reduction were inexact, the hoisted form would be approximating
  a different real number and the comparison would be measuring the reduction
  rather than the sine.
- **Two studies**, because the answer depends on what counts as the input:
    - **Study A (same-x)** — the input is one float64 `x`; all three forms
      approximate `sin(pi*x)` for that exact `x`. A like-for-like comparison of
      three routes to the same real number.
    - **Study B (kernel-faithful)** — the input is the `(frac, k)` pair the loop
      actually holds and the target is `sin(pi*(frac + k))` for the **exact**
      sum. The direct form must round that sum into a float64 first, which
      discards low bits of `frac` once `|k|` is large; the hoisted forms never
      form it. This is the error the kernel incurs.

    Study B is the harsher test of the direct form and it agrees with Study A:
    pooled medians 83 / 0.369 / 0.257 ulps for direct / `[0,1)` / `[-1/2,1/2]`,
    with the hoisted form closer at 96.06% of inputs against 1.16%. **The
    conclusion does not depend on which framing is used.**

### Sweep

45 837 inputs in four labelled groups:

| group            | inputs | what it covers                                                             |
| ---------------- | ------ | -------------------------------------------------------------------------- |
| `tap_loop`       | 29 696 | `\|k\| <= 63` (the 127-tap loop) x 232 phases spanning `[0, 1)`            |
| `large_offset`   | 6 192  | `\|k\|` from 1e1 to 7e6, where `fl(pi)*x` loses bits to argument reduction |
| `random_decades` | 9 800  | uniform random `x` over decades 1e0 to 1e7, both signs                     |
| `integer_x`      | 149    | `frac == 0` exactly, where the true value is exactly 0                     |

The phase axis is not purely random: it enumerates `2**-j` and `1 - 2**-j` for
`j = 1..53` and the dyadic rationals, so both branch boundaries of the reduction
are hit exactly rather than approached by luck.

### Metrics

Error is `|form - reference|`, differenced in high precision, reported in
absolute terms and in ulps of the reference. Ulp counts are given twice: over
all points with a non-zero reference, and over the subset with
`|reference| > 1e-3`. The distinction matters — within ~3e-4 of a zero of
`sin(pi*x)` the result's own ulp collapses while the argument's representation
error does not, so an ulp count there measures the conditioning of the zero
rather than any form's quality, and mean ulps over the unrestricted set are
dominated by those points (which is why the headline table uses the restricted
set and the median). Absolute error is the meaningful metric near the zeros, and
it tells the same story: max 2.9e-9 direct against 3.43e-16 hoisted.

## The advantage scales with the integer part

Median and max ulps at fixed `|k|`, over 300 phases in `[0.05, 0.95]` (Study A):

| `\|k\|` | `direct`        | `hoist[0,1)` | `hoist[-1/2,1/2]` |
| ------- | --------------- | ------------ | ----------------- |
| 0       | 0.377 / 10.3    | 0.377 / 10.3 | 0.310 / 1.17      |
| 1       | 1.05 / 21.6     | 0.391 / 7.91 | 0.312 / 1.09      |
| 4       | 2.59 / 48.3     | 0.401 / 7.08 | 0.298 / 1.11      |
| 16      | 10.4 / 188      | 0.335 / 11.5 | 0.259 / 1.14      |
| 63      | 36.4 / 750      | 0.367 / 9.79 | 0.304 / 1.14      |
| 1e2     | 74.8 / 1.41e3   | 0.389 / 9.01 | 0.307 / 1.14      |
| 1e3     | 615 / 1.22e4    | 0.366 / 10.5 | 0.279 / 1.06      |
| 1e4     | 5.38e3 / 1.01e5 | 0.380 / 10.3 | 0.285 / 1.48      |
| 1e5     | 7.02e4 / 1.36e6 | 0.399 / 11.0 | 0.294 / 1.20      |
| 1e6     | 5.80e5 / 1.12e7 | 0.368 / 9.23 | 0.276 / 1.05      |

The direct form's median error grows linearly with `|k|` across six decades — a
factor of 1.5e6 from `|k| = 0` to `|k| = 1e6`, matching `|x|` — while both
hoisted forms stay flat at a third of an ulp. At `|k| = 0` the direct and
`[0,1)` forms are identical by construction, which is the control: the rows
agree exactly, as they must.

This is the decomposition behind the headline number. The direct form's error is
not "sine error"; it is **argument error**, in a term the hoist deletes.

## Where the direct form wins, and why it does not rescue it

The direct form is strictly closer at 1.38% of pooled inputs. Those wins are not
scattered: in the tap regime they concentrate at **small `|x|`** (median `|x|`
4.8, 90th percentile 36) and skew hard toward `frac -> 1` (46.7% of them have
`frac > 0.99`, against 19.4% of the group as a whole). That is the `[0, 1)`
reduction's one weakness: at `frac -> 1` its argument `fl(pi)*frac` sits next to
pi, and the rounding of that product eats the leading digits of a result that is
itself near zero. When `|x|` is small there is no compensating
argument-reduction penalty on the direct form, so the direct form wins there —
by a median factor of 2.4x, occasionally by 438x.

Isolating that boundary (median ulps over `k` in `[-63, 1000]`, Study A;
selected rows from the script's full table):

| `1 - frac` | `direct` | `hoist[0,1)` | `hoist[-1/2,1/2]` |
| ---------- | -------- | ------------ | ----------------- |
| 0.5        | 0        | 0            | 0                 |
| 3.12e-2    | 125      | 16.1         | 0.118             |
| 1.95e-3    | 1.45e3   | 373          | 0.104             |
| 1.22e-4    | 2.01e4   | 1.09e3       | 0.477             |
| 7.63e-6    | 4.18e5   | 2.46e4       | 0.174             |
| 4.77e-7    | 6.77e6   | 3.05e5       | 0.407             |
| 2.98e-8    | 1.09e8   | 4.79e6       | 0.060             |
| 1.86e-9    | 1.47e9   | 3.45e8       | 0.235             |
| 1.16e-10   | 8.39e9   | 5.25e9       | 0.276             |
| 1.11e-16   | 2.21e15  | 4.42e15      | 0.276             |

Both the direct and `[0, 1)` forms degrade without bound as `x` approaches an
integer; the `[0, 1)` hoist is 4–23x better than direct through most of that
approach and only catches up (then falls marginally behind) in the last decade
or two, where both are meaningless in relative terms. The symmetric reduction
stays under an ulp throughout — it is the only form that does not have this
failure mode, because `r -> 0` there and a small argument is exactly the
well-conditioned case. Note also that at **exactly** integer `x` both hoists
return exactly 0, which is exactly right, while the direct form returns up to
2.23e-10.

At more than half of the points where `direct` beats `hoist[0,1)` in the tap
regime (328 of 537), `hoist[-1/2,1/2]` beats `direct` anyway. **Choosing the
symmetric reduction removes the exception rather than trading against it.**

## Anchors

Per the discipline of not quoting a number that has never been checked against
something external:

1. **Analytic prediction of the direct form's error.**
   `pi - fl(pi) = 1.2246467991473532e-16` — a constant independent of this
   study; it is also what libm returns for `sin(float64(pi))`. Predicting the
   direct form's error as `|cos(pi*x)| * |x| * (pi - fl(pi))` gives
   measured/predicted with **median 1.10** (random decades) and **1.18** (large
   offsets), 10th–90th percentile 0.18–2.7. The spread is the `|cos|` factor and
   the product rounding the prediction omits; the median near 1 says the
   mechanism is identified, not merely correlated.
2. **First-order error bound, all three forms.** Bounding the argument
   displacement by `|a| * (pi - fl(pi)) + 0.5 ulp(pi*|a|)`, converting through
   `|cos(pi*a)|`, and adding 1 ulp for libm gives a bound satisfied at **45 688
   of 45 688** Study A points in the three ulp-bearing groups, with max
   measured/bound = 1.00. In Study B the direct form exceeds this bound at ~2
   400 points (max ratio 2.01) — correctly, because Study B adds the rounding of
   `frac + k`, which the bound deliberately omits. The bound holds for both
   hoisted forms in both studies.
3. **Independent sine implementation.** numpy dispatches `np.sin` to a SIMD
   kernel that is not the platform libm's scalar `sin`. On all 29 696 tap-regime
   arguments, for all three forms, the two implementations returned
   **bit-identical** results (max gap 0 ulps). The ranking is therefore not an
   artefact of numpy's vectorised sine.
4. **Reference validated** at two precisions and by two algebraically distinct
   routes (see above): agreement to 2.6e-51 relative.

### What is _not_ anchored here

Stated explicitly, because these are the gaps a reader would otherwise assume
were covered:

- **The speedup is not measured in this study.** This is an accuracy measurement
  only. It says nothing about how much time the hoist saves.
- **End-to-end effect on resampled output is not measured.** This study measures
  the accuracy of the tap value `sin(pi*x)`. A resampled sample is a normalised
  weighted sum of 127 taps, and per-tap errors can cancel or accumulate in that
  sum. A difference measured on resampled output is therefore a _different
  quantity_ from the per-tap differences reported here, and the two need not
  agree in magnitude. Relatedly, the disagreement _between_ the two forms in the
  tap regime has median 111 ulps here — but inter-form distance is not an error
  bar for either form, and on its own says nothing about which is closer to the
  truth.
- **The full tap weight was not measured.** The kernel's weight is
  `sin(pi*x)/(pi*x) * window(x)`, not `sin(pi*x)`. Relative error passes through
  the division essentially unchanged, so the ranking should carry, but the
  composite was not measured.
- **One platform, one libm.** Measured with numpy 2.5.2 / Python 3.14.7 on
  x86-64 glibc. The dominant term is argument-side, i.e. a property of `fl(pi)`
  and `|x|` rather than of any sine implementation, so the ranking should carry
  to other libms and to GPU backends — but sub-ulp behaviour will differ and was
  not measured there. A GPU kernel in float32 would shift every number in this
  report; nothing here speaks to float32.

## Reproducing it

```console
uv run --with mpmath python scripts/sin_hoist_accuracy.py
```

`mpmath` is supplied on the fly by `--with` and is deliberately _not_ added to
the project's dependencies — this is a standalone study, not part of the
package.

The script depends only on numpy and mpmath, writes its full report to stdout as
markdown, and takes ~4 s wall-clock. The sweep's random axes are seeded
(`--seed`, default 20260823), so the run is reproducible; `--skip-checks` drops
the two verification passes. Every number quoted in this document comes from
that output.
