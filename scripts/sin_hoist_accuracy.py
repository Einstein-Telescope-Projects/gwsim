"""Measure which float64 evaluation of ``sin(pi*x)`` is the most accurate.

A windowed-sinc resample tap loop evaluates ``sin(pi*x)`` once per tap, where the tap
argument is ``x = k + frac``: ``k`` the integer tap offset and ``frac`` the fractional
sample phase. Because ``sin(pi*(k + f)) = (-1)**k * sin(pi*f)`` exactly in real
arithmetic, the transcendental can be hoisted out of the loop -- evaluated once, then
turned into each tap's value by a sign flip, which is exact. In floating point the forms
are not bit-identical, so "which one is closer to the true value" is an empirical
question, and that is what this script settles.

Three float64 forms are compared against an mpmath reference carried at 50 decimal
digits (``pi`` included):

1. ``direct``          -- ``sin(pi*x)``, one transcendental per tap.
2. ``hoist[0,1)``      -- ``(-1)**floor(x) * sin(pi*frac)`` with ``frac = x - floor(x)``
   in ``[0, 1)``: the form under consideration.
3. ``hoist[-1/2,1/2]`` -- ``(-1)**n * sin(pi*r)`` with ``n = round(x)`` and
   ``r = x - n`` in ``[-1/2, 1/2]``: the same hoist, reduced symmetrically instead. It
   saves exactly as many transcendentals, since ``r`` is also loop-invariant and the
   per-tap sign is still just a parity flip.

The third form is included because the second has a failure mode the first does not:
``frac -> 1`` puts the reduced argument next to ``pi``, where rounding the product
``fl(pi)*frac`` costs the result its leading digits. Symmetric reduction keeps the
argument within ``pi/2`` of zero and cannot hit that.

Two studies are reported, because the honest comparison depends on what the input is:

* **Study A (same-x)** -- the input is a single float64 ``x``. Every form approximates
  ``sin(pi*x)`` for that exact ``x``; the reductions ``x - floor(x)`` and ``x - round(x)``
  are exact in float64 (the script verifies this in high precision at every input rather
  than asserting it), so all three routes lead to the same real number.
* **Study B (kernel-faithful)** -- the input is the pair ``(frac, k)`` the loop actually
  holds, and the target is ``sin(pi*(frac + k))`` for the *exact* sum. The direct form
  has to round that sum into a float64 first, discarding low bits of ``frac`` once
  ``|k|`` is large; the hoisted forms never form it. This is the error the kernel incurs.

Run it with mpmath supplied on the fly, so it need not become a project dependency::

    uv run --with mpmath python scripts/sin_hoist_accuracy.py

The report is written to stdout as markdown.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from mpmath import mp, mpf

#: Working precision of the reference. 50 decimal digits is ~166 bits, so a reference
#: value carries ~113 bits beyond float64's 53 and its own error is negligible next to
#: the ~1e-16 effects under study.
REFERENCE_DPS = 50

#: Precision of the independent cross-check of the reference (see `validate_reference`).
CROSSCHECK_DPS = 120

#: pi - float64(pi), the leading error term of any form that multiplies its argument by
#: `fl(pi)`. Independent of this script: it is the value libm returns for
#: ``sin(float64(pi))``, since ``sin(fl(pi)) = sin(pi - (pi - fl(pi))) ~ pi - fl(pi)``.
PI_REPRESENTATION_DEFECT = 1.2246467991473532e-16

#: Below this reference magnitude, ``sin(pi*x)`` sits so close to one of its zeros that
#: an ulp count says more about the conditioning of the zero than about any form: the
#: result's own ulp shrinks with it while the argument's representation error does not.
#: Statistics are reported both over everything and over the subset above this floor.
WELL_CONDITIONED_FLOOR = 1e-3

#: The `frac -> 1` end of the unit-interval reduction, where its reduced argument sits
#: next to pi. Used to report how much of the direct form's occasional advantage lives there.
NEAR_ONE_FRACTION = 0.99

DIRECT = "direct"
HOIST_UNIT = "hoist[0,1)"
HOIST_SYMMETRIC = "hoist[-1/2,1/2]"
FORMS = (DIRECT, HOIST_UNIT, HOIST_SYMMETRIC)


@dataclass(frozen=True)
class Group:
    """One labelled block of the sweep, held as the ``(frac, k)`` pairs the loop sees."""

    name: str
    description: str
    frac: np.ndarray
    k: np.ndarray

    def __post_init__(self) -> None:
        if self.frac.shape != self.k.shape:
            raise ValueError(f"group {self.name}: frac and k must have the same shape")
        if self.frac.size == 0:
            raise ValueError(f"group {self.name}: empty group")
        if not np.all((self.frac >= 0.0) & (self.frac < 1.0)):
            raise ValueError(f"group {self.name}: frac must lie in [0, 1)")
        if not np.all(self.k == np.floor(self.k)):
            raise ValueError(f"group {self.name}: k must be integral")


def _fractional_phases(rng: np.random.Generator, count: int) -> np.ndarray:
    """Fractional phases spanning the open interval plus the values that break things.

    Uniform draws alone never land on a boundary, and the boundaries are where the
    hoisted forms' branch structure lives, so they are enumerated explicitly.
    """
    exponents = np.arange(1, 54)
    return np.unique(
        np.concatenate(
            [
                rng.uniform(0.0, 1.0, size=count),
                # dyadic rationals: pi*frac is a benign product, isolating libm's own error
                np.array([0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875]),
                # frac -> 0 boundary, down to where frac + k can no longer see frac at all
                2.0**-exponents,
                # frac -> 1 boundary; nextafter keeps every value strictly below 1
                np.nextafter(1.0 - 2.0**-exponents, 0.0),
                np.array([np.nextafter(1.0, 0.0), 0.0]),
            ]
        )
    )


def build_groups(rng: np.random.Generator) -> list[Group]:
    """Assemble the sweep as labelled ``(frac, k)`` blocks."""
    groups: list[Group] = []
    decades = np.array([1e1, 1e2, 1e3, 1e4, 1e5, 1e6])

    # --- the regime the kernel is actually in: a 127-tap loop, so |k| <= 63 ---------
    phases = _fractional_phases(rng, 120)
    taps = np.arange(-63, 65, dtype=np.float64)
    frac_grid, k_grid = np.meshgrid(phases, taps, indexing="ij")
    groups.append(
        Group(
            name="tap_loop",
            description="127-tap resample loop: |k| <= 63, frac spanning [0, 1) with boundaries",
            frac=frac_grid.ravel(),
            k=k_grid.ravel(),
        )
    )

    # --- large |x|, where pi*x argument reduction is the dominant error -------------
    offsets = np.unique(np.concatenate([decades, decades * 3, decades * 7])).astype(np.float64)
    offsets = np.concatenate([offsets, -offsets])
    phases_large = _fractional_phases(rng, 60)
    frac_grid, k_grid = np.meshgrid(phases_large, offsets, indexing="ij")
    groups.append(
        Group(
            name="large_offset",
            description="|k| from 1e1 to 7e6, where fl(pi)*x loses bits to argument reduction",
            frac=frac_grid.ravel(),
            k=k_grid.ravel(),
        )
    )

    # --- random x over decades, decomposed back into (frac, k) ---------------------
    magnitudes: list[np.ndarray] = []
    for exponent in range(7):
        sample = rng.uniform(10.0**exponent, 10.0 ** (exponent + 1), size=700)
        magnitudes.append(sample)
        magnitudes.append(-sample)
    x_random = np.concatenate(magnitudes)
    k_random = np.floor(x_random)
    groups.append(
        Group(
            name="random_decades",
            description="uniform random x over decades 1e0 to 1e7, both signs",
            frac=x_random - k_random,
            k=k_random,
        )
    )

    # --- exact integers: the true value is exactly 0, so any output is pure error ---
    integers = np.unique(np.concatenate([np.arange(0, 65, dtype=np.float64), decades, decades * 3]))
    integers = np.concatenate([integers, -integers[1:]])
    groups.append(
        Group(
            name="integer_x",
            description="frac == 0 exactly: sin(pi*x) is exactly 0, so any output is pure error",
            frac=np.zeros_like(integers),
            k=integers,
        )
    )

    return groups


def _sign_from_parity(offset: np.ndarray) -> np.ndarray:
    """``(-1)**offset`` for integral float64 offsets, exact for ``|offset| < 2**53``."""
    return np.where(np.abs(offset) % 2.0 == 0.0, 1.0, -1.0)


def high_precision_reference(offset: np.ndarray, reduced: np.ndarray, dps: int = REFERENCE_DPS) -> list[mpf]:
    """``sin(pi*(offset + reduced))`` at ``dps`` digits, from the exact reduced argument.

    ``sin(pi*(n + r)) = (-1)**n * sin(pi*r)`` is an identity in the reals, so using it
    costs no accuracy, and it buys two things a direct high-precision ``sin(pi*x)`` does
    not: the result is *exactly* zero when ``reduced`` is zero, and no working digits are
    spent on an argument of magnitude up to ``pi*7e6``. `validate_reference` checks the
    result against the unreduced computation at much higher precision, so the convenience
    is measured rather than assumed.
    """
    sign = _sign_from_parity(offset)
    with mp.workdps(dps):
        pi = mp.pi
        return [int(s) * mp.sin(pi * mpf(float(r))) for s, r in zip(sign, reduced, strict=True)]


def validate_reference(x_exact: list[mpf], reference: list[mpf]) -> tuple[float, int]:
    """Cross-check the reduced reference against an unreduced one at ``CROSSCHECK_DPS``.

    Returns the largest relative discrepancy and the number of points compared. Points
    whose reference is exactly zero are skipped: the unreduced computation cannot return
    zero there (``pi`` is irrational at any finite precision), so a relative comparison is
    undefined -- and ``sin(k*pi) = 0`` needs no numerical confirmation.
    """
    worst = 0.0
    compared = 0
    with mp.workdps(CROSSCHECK_DPS):
        pi = mp.pi
        for exact, reduced in zip(x_exact, reference, strict=True):
            if reduced == 0:
                continue
            unreduced = mp.sin(pi * exact)
            worst = max(worst, float(abs(unreduced - reduced) / abs(unreduced)))
            compared += 1
    return worst, compared


def _absolute_errors(values: np.ndarray, reference: list[mpf]) -> np.ndarray:
    with mp.workdps(REFERENCE_DPS):
        return np.array([float(abs(mpf(float(v)) - r)) for v, r in zip(values, reference, strict=True)])


@dataclass
class Evaluated:
    """Every float64 form and the high-precision reference, for one study of one group."""

    reference: np.ndarray  # reference rounded to float64; sets the ulp scale
    ulp: np.ndarray  # ulp of the reference magnitude
    values: dict[str, np.ndarray]  # form name -> float64 result
    errors: dict[str, np.ndarray]  # form name -> |result - reference|, differenced exactly
    arguments: dict[str, np.ndarray]  # form name -> the argument that form fed to sin
    x: np.ndarray  # the float64 argument the direct form evaluated
    k: np.ndarray  # the integer part whose parity supplied the sign


def _assemble(
    values: dict[str, np.ndarray],
    arguments: dict[str, np.ndarray],
    reference: list[mpf],
    x: np.ndarray,
    k: np.ndarray,
) -> Evaluated:
    reference_f64 = np.array([float(r) for r in reference])
    return Evaluated(
        reference=reference_f64,
        ulp=np.spacing(np.abs(reference_f64)),
        values=values,
        errors={name: _absolute_errors(value, reference) for name, value in values.items()},
        arguments=arguments,
        x=x,
        k=k,
    )


def evaluate_study_a(group: Group) -> Evaluated:
    """Every form as an approximation of ``sin(pi*x)`` for one float64 ``x``."""
    x = group.frac + group.k  # a float64 value; need not equal frac + k in the reals
    floor = np.floor(x)
    unit = x - floor  # exact in float64; checked by verify_reductions_are_exact
    nearest = np.round(x)
    symmetric = x - nearest  # likewise exact
    values = {
        DIRECT: np.sin(np.pi * x),
        HOIST_UNIT: _sign_from_parity(floor) * np.sin(np.pi * unit),
        HOIST_SYMMETRIC: _sign_from_parity(nearest) * np.sin(np.pi * symmetric),
    }
    arguments = {DIRECT: x, HOIST_UNIT: unit, HOIST_SYMMETRIC: symmetric}
    reference = high_precision_reference(nearest, symmetric)
    return _assemble(values, arguments, reference, x, floor)


def evaluate_study_b(group: Group) -> Evaluated:
    """Every form as an approximation of ``sin(pi*(frac + k))`` for the *exact* sum.

    This is what the tap loop is asked for: ``frac`` and ``k`` arrive as separate inputs
    and the intended argument is their exact sum. The direct form has to round that sum
    into a float64 before it can call ``sin``; the hoisted forms never form it.
    """
    x = group.frac + group.k
    nearest_frac = np.round(group.frac)  # 0.0 or 1.0
    symmetric = group.frac - nearest_frac  # exact: Sterbenz applies for frac in [1/2, 1)
    nearest = group.k + nearest_frac
    values = {
        DIRECT: np.sin(np.pi * x),
        HOIST_UNIT: _sign_from_parity(group.k) * np.sin(np.pi * group.frac),
        HOIST_SYMMETRIC: _sign_from_parity(nearest) * np.sin(np.pi * symmetric),
    }
    arguments = {DIRECT: x, HOIST_UNIT: group.frac, HOIST_SYMMETRIC: symmetric}
    reference = high_precision_reference(nearest, symmetric)
    return _assemble(values, arguments, reference, x, group.k)


def verify_reductions_are_exact(groups: list[Group]) -> tuple[int, int]:
    """Check that both argument reductions are exact in float64, at every sweep input.

    The hoisted forms' fairness rests on this: were a subtraction inexact, that form would
    be evaluating a different real number and the comparison would be measuring the
    reduction rather than the sine. Instead of arguing it in a comment, ``x - floor(x)``
    and ``x - round(x)`` are both checked in high precision at every input the sweep
    visits, and separately for Study B's ``frac - round(frac)``.
    """
    violations = 0
    total = 0
    with mp.workdps(REFERENCE_DPS):
        for group in groups:
            x = group.frac + group.k
            reductions = (
                (x, np.floor(x)),
                (x, np.round(x)),
                (group.frac, np.round(group.frac)),
            )
            for value, integral in reductions:
                remainder = value - integral
                for value_i, integral_i, remainder_i in zip(value, integral, remainder, strict=True):
                    total += 1
                    if mpf(float(value_i)) - mpf(float(integral_i)) != mpf(float(remainder_i)):
                        violations += 1
    return violations, total


def _ulp_errors(evaluated: Evaluated, form: str, floor: float = 0.0) -> np.ndarray:
    """One form's error in ulps of the reference, over points whose reference beats ``floor``."""
    keep = _conditioned_mask(evaluated, floor)
    return evaluated.errors[form][keep] / evaluated.ulp[keep]


def _conditioned_mask(evaluated: Evaluated, floor: float = 0.0) -> np.ndarray:
    return (evaluated.ulp > 0.0) & (np.abs(evaluated.reference) > floor)


def _fmt(value: float) -> str:
    if value == 0.0:
        return "0"
    return f"{value:.3g}"


def _table(header: str, rows: list[tuple[str, dict[str, float]]]) -> None:
    print(f"| {header} | " + " | ".join(f"`{form}`" for form in FORMS) + " |")
    print("| --- " * (len(FORMS) + 1) + "|")
    for label, values in rows:
        print(f"| {label} | " + " | ".join(_fmt(values[form]) for form in FORMS) + " |")


def report_group(group: Group, evaluated: Evaluated) -> None:
    """Emit one group's per-form statistics as markdown."""
    all_ulps = {form: _ulp_errors(evaluated, form) for form in FORMS}
    conditioned = {form: _ulp_errors(evaluated, form, WELL_CONDITIONED_FLOOR) for form in FORMS}
    count = evaluated.x.size

    print(f"\n#### `{group.name}` -- {group.description}\n")
    print(f"Points: {count} ({all_ulps[DIRECT].size} with a non-zero reference, so ulps are defined).\n")

    rows: list[tuple[str, dict[str, float]]] = [
        ("max absolute error", {form: float(np.max(evaluated.errors[form])) for form in FORMS})
    ]
    for label, sample in (("all", all_ulps), (f"reference > {WELL_CONDITIONED_FLOOR:g}", conditioned)):
        if not sample[DIRECT].size:
            continue
        for statistic, reduce_fn in (("max", np.max), ("mean", np.mean), ("median", np.median)):
            rows.append(
                (f"{statistic} error, ulps ({label})", {form: float(reduce_fn(sample[form])) for form in FORMS})
            )
    _table("metric", rows)

    if conditioned[DIRECT].size:
        print(
            f"\nThe `reference > {WELL_CONDITIONED_FLOOR:g}` rows cover {conditioned[DIRECT].size} of "
            f"{all_ulps[DIRECT].size} points; the rest sit within ~3e-4 of a zero of `sin(pi*x)`, where the "
            "result's own ulp collapses, so an ulp count there measures the conditioning of the zero rather "
            "than any form. Absolute error is the meaningful metric for those."
        )

    for challenger in (HOIST_UNIT, HOIST_SYMMETRIC):
        _report_pairwise(evaluated, DIRECT, challenger)
    _report_pairwise(evaluated, HOIST_UNIT, HOIST_SYMMETRIC)
    _report_worst_cases(evaluated)


def _report_pairwise(evaluated: Evaluated, left: str, right: str) -> None:
    """Win counts and win sizes for one pair of forms."""
    err_left = evaluated.errors[left]
    err_right = evaluated.errors[right]
    count = err_left.size
    left_wins = int(np.sum(err_left < err_right))
    right_wins = int(np.sum(err_right < err_left))
    print(
        f"\n`{left}` vs `{right}`: `{left}` closer at {left_wins} points "
        f"({100.0 * left_wins / count:.2f}%), `{right}` closer at {right_wins} ({100.0 * right_wins / count:.2f}%), "
        f"identical at {count - left_wins - right_wins}."
    )
    conditioned = _conditioned_mask(evaluated, WELL_CONDITIONED_FLOOR)
    for winner, loser in ((left, right), (right, left)):
        won = conditioned & (evaluated.errors[winner] < evaluated.errors[loser])
        if not np.any(won):
            continue
        margin = (evaluated.errors[loser][won] - evaluated.errors[winner][won]) / evaluated.ulp[won]
        winner_error = evaluated.errors[winner][won]
        positive = winner_error > 0.0
        ratio = evaluated.errors[loser][won][positive] / winner_error[positive]
        ratio_text = (
            f"error ratio median {_fmt(float(np.median(ratio)))}x, max {_fmt(float(np.max(ratio)))}x"
            if ratio.size
            else f"`{winner}` was exact at all of them"
        )
        print(
            f"  Where `{winner}` wins and the reference exceeds {WELL_CONDITIONED_FLOOR:g} "
            f"({int(np.sum(won))} points): margin median {_fmt(float(np.median(margin)))} ulps, "
            f"max {_fmt(float(np.max(margin)))} ulps; {ratio_text}."
        )


def _report_worst_cases(evaluated: Evaluated) -> None:
    """Name the inputs at which each form is at its worst, in absolute error and in ulps."""
    conditioned = _conditioned_mask(evaluated, WELL_CONDITIONED_FLOOR)
    print()
    for form in FORMS:
        worst = int(np.argmax(evaluated.errors[form]))
        others = " / ".join(
            f"`{other}` {_fmt(float(evaluated.errors[other][worst]))}" for other in FORMS if other != form
        )
        print(
            f"- `{form}` worst absolute error {_fmt(float(evaluated.errors[form][worst]))} at "
            f"x = {evaluated.x[worst]!r} (k = {evaluated.k[worst]:.0f}, reduced argument "
            f"{evaluated.arguments[form][worst]!r}); there {others}."
        )
    if not np.any(conditioned):
        return
    for form in FORMS:
        ulps = evaluated.errors[form][conditioned] / evaluated.ulp[conditioned]
        worst = int(np.argmax(ulps))
        others = " / ".join(
            f"`{other}` {_fmt(float(evaluated.errors[other][conditioned][worst] / evaluated.ulp[conditioned][worst]))}"
            for other in FORMS
            if other != form
        )
        print(
            f"- `{form}` worst ulp count {_fmt(float(ulps[worst]))} ulps at "
            f"x = {evaluated.x[conditioned][worst]!r} (reference > {WELL_CONDITIONED_FLOOR:g}); "
            f"there {others} ulps."
        )


def report_libm_crosscheck(evaluated: Evaluated) -> None:
    """Check numpy's vectorised sine against the scalar libm sine, per form.

    numpy dispatches `np.sin` to a SIMD kernel that is not the same code as the platform
    libm's scalar `sin`, so a result attributed here to one form could in principle be an
    artefact of that kernel. Comparing the two implementations on identical arguments
    separates "this form feeds the sine a better argument" -- the claim under test -- from
    "this sine implementation happens to round differently".
    """
    for form in FORMS:
        argument = np.pi * evaluated.arguments[form]
        scalar = np.array([math.sin(float(value)) for value in argument])
        vector = np.sin(argument)
        scale = np.spacing(np.abs(vector))
        differing = int(np.sum(scalar != vector))
        gap = np.abs(scalar - vector)[scale > 0.0] / scale[scale > 0.0]
        print(
            f"\nnumpy vs scalar libm, `{form}`: {differing} of {argument.size} arguments give a different "
            f"float64; max gap {_fmt(float(np.max(gap)))} ulps, median {_fmt(float(np.median(gap)))} ulps."
        )


def report_direct_win_profile(evaluated: Evaluated) -> None:
    """Characterise the inputs where the direct form beats the unit-interval hoist.

    A win count says the direct form is occasionally better; it does not say when, and
    "when" is what decides whether the exception matters. The hoist's exposure is
    ``frac -> 1``, where its reduced argument sits next to pi and the product rounding eats
    the leading digits, so that is the axis reported.
    """
    won = evaluated.errors[DIRECT] < evaluated.errors[HOIST_UNIT]
    if not np.any(won):
        print("\nThe direct form is never strictly closer than `hoist[0,1)` in this group.")
        return
    unit = evaluated.arguments[HOIST_UNIT]
    distance_to_one = 1.0 - unit[won]
    magnitude = np.abs(evaluated.x[won])
    near_one = float(np.mean(unit[won] > NEAR_ONE_FRACTION))
    print(
        f"\nWhere `direct` beats `hoist[0,1)` ({int(np.sum(won))} points): |x| has median "
        f"{_fmt(float(np.median(magnitude)))} and 90th percentile {_fmt(float(np.percentile(magnitude, 90)))}; "
        f"1 - frac has median {_fmt(float(np.median(distance_to_one)))}; {100.0 * near_one:.1f}% of them have "
        f"frac > {NEAR_ONE_FRACTION}, against {100.0 * float(np.mean(unit > NEAR_ONE_FRACTION)):.1f}% of the group as a whole. "
        f"`hoist[-1/2,1/2]` beats `direct` at {int(np.sum(evaluated.errors[HOIST_SYMMETRIC][won] < evaluated.errors[DIRECT][won]))} "
        "of those same points."
    )


def report_inter_form_difference(evaluated: Evaluated) -> None:
    """How far apart two forms are from *each other* -- the originally quoted figure."""
    keep = _conditioned_mask(evaluated)
    for left, right in ((DIRECT, HOIST_UNIT), (DIRECT, HOIST_SYMMETRIC)):
        difference = np.abs(evaluated.values[left] - evaluated.values[right])[keep] / evaluated.ulp[keep]
        print(
            f"\nDisagreement between `{left}` and `{right}` themselves: median "
            f"{_fmt(float(np.median(difference)))} ulps, 90th percentile "
            f"{_fmt(float(np.percentile(difference, 90)))} ulps, max {_fmt(float(np.max(difference)))} ulps "
            f"over {difference.size} points. This is a distance, not an error: on its own it says nothing "
            "about which form is closer to the truth."
        )


def report_error_bound(evaluated: Evaluated) -> None:
    """Check every form against a first-order error bound derived independently.

    Evaluating ``sin(fl(fl(pi)*a))`` displaces the argument from ``pi*a`` by at most
    ``|a| * (pi - fl(pi))`` (the constant is short of pi) plus ``0.5 ulp(pi*|a|)`` (the
    product is rounded); the sine turns that into ``|cos(pi*a)|`` times the displacement,
    and libm adds at most 1 ulp of the result. Only ``|a|`` differs between the forms --
    ``x`` for the direct form against a reduced argument of magnitude at most 1 -- which is
    the whole of the hoist's accuracy advantage, and it scales with ``|x|``.
    """
    result_ulp = np.spacing(np.abs(evaluated.reference))
    for form in FORMS:
        argument = evaluated.arguments[form]
        magnitude = np.abs(argument)
        displacement = magnitude * PI_REPRESENTATION_DEFECT + 0.5 * np.spacing(np.pi * magnitude)
        bound = np.abs(np.cos(np.pi * argument)) * displacement + result_ulp
        within = evaluated.errors[form] <= bound
        usable = bound > 0.0
        ratio = evaluated.errors[form][usable] / bound[usable]
        print(
            f"\nFirst-order bound, `{form}`: satisfied at {int(np.sum(within))} of {evaluated.errors[form].size} "
            f"points; measured / bound has median {_fmt(float(np.median(ratio)))} and "
            f"max {_fmt(float(np.max(ratio)))}."
        )


def report_analytic_anchor(evaluated: Evaluated) -> None:
    """Anchor the direct form's error to a closed-form prediction, independent of the sweep.

    ``fl(pi)`` falls short of pi by ``PI_REPRESENTATION_DEFECT``, so the direct form
    evaluates the sine at an argument displaced by about ``x * (pi - fl(pi))``. To first
    order that displaces the result by ``|cos(pi*x)| * |x| * (pi - fl(pi))``.
    """
    predicted = np.abs(np.cos(np.pi * evaluated.x)) * np.abs(evaluated.x) * PI_REPRESENTATION_DEFECT
    keep = predicted > 0.0
    ratio = evaluated.errors[DIRECT][keep] / predicted[keep]
    print(
        "\nAnalytic anchor for the direct form's error (prediction "
        f"|cos(pi*x)| * |x| * (pi - fl(pi)), with pi - fl(pi) = {PI_REPRESENTATION_DEFECT:.6e}): "
        f"measured / predicted has median {_fmt(float(np.median(ratio)))}, "
        f"10th percentile {_fmt(float(np.percentile(ratio, 10)))}, "
        f"90th percentile {_fmt(float(np.percentile(ratio, 90)))} over {ratio.size} points."
    )


def report_pooled(label: str, evaluated_groups: list[Evaluated]) -> None:
    """Pool every group of one study into the headline numbers."""
    pooled = {form: np.concatenate([e.errors[form] for e in evaluated_groups]) for form in FORMS}
    conditioned = {
        form: np.concatenate([_ulp_errors(e, form, WELL_CONDITIONED_FLOOR) for e in evaluated_groups]) for form in FORMS
    }
    count = pooled[DIRECT].size

    print(f"\n### Pooled over Study {label}\n")
    _table(
        "pooled metric",
        [
            ("max absolute error", {form: float(np.max(pooled[form])) for form in FORMS}),
            (
                f"max error, ulps (reference > {WELL_CONDITIONED_FLOOR:g})",
                {form: float(np.max(conditioned[form])) for form in FORMS},
            ),
            (
                f"mean error, ulps (reference > {WELL_CONDITIONED_FLOOR:g})",
                {form: float(np.mean(conditioned[form])) for form in FORMS},
            ),
            (
                f"median error, ulps (reference > {WELL_CONDITIONED_FLOOR:g})",
                {form: float(np.median(conditioned[form])) for form in FORMS},
            ),
        ],
    )
    print(f"\nPooled over {count} inputs ({conditioned[DIRECT].size} of them well-conditioned):\n")
    for left, right in ((DIRECT, HOIST_UNIT), (DIRECT, HOIST_SYMMETRIC), (HOIST_UNIT, HOIST_SYMMETRIC)):
        left_wins = int(np.sum(pooled[left] < pooled[right]))
        right_wins = int(np.sum(pooled[right] < pooled[left]))
        print(
            f"- `{left}` closer at {left_wins} points ({100.0 * left_wins / count:.2f}%), "
            f"`{right}` closer at {right_wins} ({100.0 * right_wins / count:.2f}%), "
            f"identical at {count - left_wins - right_wins}."
        )


def report_offset_scaling(rng: np.random.Generator, evaluate: Callable[[Group], Evaluated]) -> None:
    """Show how each form's error grows with ``|k|`` -- the factor that decides the question."""
    phases = rng.uniform(0.05, 0.95, size=300)
    print("\n| \\|k\\| | " + " | ".join(f"median / max ulps, `{form}`" for form in FORMS) + " |")
    print("| --- " * (len(FORMS) + 1) + "|")
    for magnitude in (0.0, 1.0, 4.0, 16.0, 63.0, 1e2, 1e3, 1e4, 1e5, 1e6):
        group = Group(
            name=f"scaling_k{magnitude:.0f}",
            description="",
            frac=phases,
            k=np.full_like(phases, magnitude),
        )
        evaluated = evaluate(group)
        cells = []
        for form in FORMS:
            ulps = _ulp_errors(evaluated, form)
            cells.append(f"{_fmt(float(np.median(ulps)))} / {_fmt(float(np.max(ulps)))}")
        print(f"| {magnitude:.0f} | " + " | ".join(cells) + " |")


def report_frac_boundary() -> None:
    """Isolate the `frac -> 1` boundary, where the two reductions part company."""
    distances = 2.0 ** -np.arange(1, 54, 4, dtype=np.float64)
    print("\n| 1 - frac | " + " | ".join(f"median ulps, `{form}`" for form in FORMS) + " |")
    print("| --- " * (len(FORMS) + 1) + "|")
    for distance in distances:
        frac = np.full(9, 1.0 - distance)
        group = Group(
            name="boundary",
            description="",
            frac=frac,
            k=np.array([-63.0, -16.0, -1.0, 0.0, 1.0, 4.0, 16.0, 63.0, 1000.0]),
        )
        evaluated = evaluate_study_a(group)
        cells = [_fmt(float(np.median(_ulp_errors(evaluated, form)))) for form in FORMS]
        print(f"| {distance:.3g} | " + " | ".join(cells) + " |")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=20260823, help="seed for the random sweep axes")
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="skip the reduction-exactness check and the reference cross-check (the slow parts)",
    )
    arguments = parser.parse_args(argv)

    mp.dps = REFERENCE_DPS
    rng = np.random.default_rng(arguments.seed)
    groups = build_groups(rng)
    total = sum(group.frac.size for group in groups)

    print("# Accuracy of three float64 evaluations of `sin(pi*x)`\n")
    print(
        f"Reference: mpmath at {REFERENCE_DPS} decimal digits, `pi` carried at the same precision. "
        f"numpy {np.__version__}, Python {sys.version.split()[0]}, seed {arguments.seed}.\n"
    )
    print(f"Sweep: {total} inputs across {len(groups)} groups.\n")

    if arguments.skip_checks:
        print("Reduction-exactness check and reference cross-check: **skipped** by request.\n")
    else:
        violations, checked = verify_reductions_are_exact(groups)
        print(
            f"Reduction exactness: `x - floor(x)`, `x - round(x)` and `frac - round(frac)` were exact at "
            f"**{checked - violations} of {checked}** float64 inputs (violations: {violations}). Every form "
            "therefore evaluates the same real number, so the comparison is like-for-like.\n"
        )
        with mp.workdps(REFERENCE_DPS):
            for group in groups:
                nearest = np.round(group.frac)
                reference = high_precision_reference(group.k + nearest, group.frac - nearest)
                exact = [mpf(float(f)) + mpf(float(offset)) for f, offset in zip(group.frac, group.k, strict=True)]
                worst, compared = validate_reference(exact, reference)
                print(
                    f"Reference cross-check, `{group.name}`: the reduced reference at {REFERENCE_DPS} digits "
                    f"agrees with an unreduced `sin(pi*x)` at {CROSSCHECK_DPS} digits to a maximum relative "
                    f"discrepancy of {_fmt(worst)} over {compared} points."
                )
        print()

    for label, evaluate in (("A (same-x)", evaluate_study_a), ("B (kernel-faithful)", evaluate_study_b)):
        print(f"\n## Study {label}\n")
        evaluated_groups = []
        for group in groups:
            evaluated = evaluate(group)
            evaluated_groups.append(evaluated)
            report_group(group, evaluated)
            if group.name == "tap_loop":
                report_inter_form_difference(evaluated)
                report_libm_crosscheck(evaluated)
            if group.name in {"tap_loop", "large_offset", "random_decades"}:
                report_direct_win_profile(evaluated)
                report_error_bound(evaluated)
            if group.name in {"large_offset", "random_decades"}:
                report_analytic_anchor(evaluated)
        report_pooled(label, evaluated_groups)

        print(f"\n### Error growth with the integer part -- Study {label}\n")
        report_offset_scaling(rng, evaluate)

    print("\n## The `frac -> 1` boundary (Study A, median over k in [-63, 1000])\n")
    report_frac_boundary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
