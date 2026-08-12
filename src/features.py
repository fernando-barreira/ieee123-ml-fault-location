"""Physics-informed feature engineering from raw PMU phasors.

The raw dataset holds, for each PMU and each phase, four phasors: voltage and
current, before and during the fault.  This module turns them into quantities a
protection engineer would recognise, because those are the quantities that
actually carry information about *where* the fault is.

Feature families and what each one is for
------------------------------------------

**Superimposed (delta) phasors** — :math:`\\Delta\\tilde V`, :math:`\\Delta\\tilde I`

    .. math:: \\Delta \\tilde X = \\tilde X_\\text{fault} - \\tilde X_\\text{pre}

    By superposition, the faulted network is the pre-fault network plus a
    "pure fault" network driven only by a voltage source at the fault point.
    The delta phasors are the response of that second network, so they are
    largely free of load current and of the pre-fault power transfer.  This is
    what lets one model cover a ±30 % load range.  Computed in rectangular
    coordinates (:func:`phasor_delta`) because subtracting magnitudes and
    angles separately is not phasor subtraction.

**Ratios** — ``ratioV``, ``ratioI`` and their logs

    The per-unit voltage sag and current surge.  Scale-free, so they transfer
    across operating points better than absolute magnitudes.  The logarithm
    linearises the multiplicative relationship between distance and sag.

**Apparent impedance** — ``Z_est``, ``log_absZ``, ``sin/cos`` of its angle

    .. math:: \\tilde Z = \\Delta \\tilde V / \\Delta \\tilde I

    The superimposed-impedance estimate is the core of distance protection: for
    a fault on a homogeneous feeder its magnitude grows roughly with the
    line length between the PMU and the fault, and its angle approaches the
    line impedance angle for a bolted fault, rotating towards 0° as the fault
    resistance grows.  A PMU therefore contributes a "how far along which
    direction" estimate, and the classifier fuses several of them.

**Symmetrical components of the deltas** — :math:`\\Delta V_{0,1,2}`,
:math:`\\Delta I_{0,1,2}`

    Fortescue's transformation with :math:`a = e^{j2\\pi/3}`:

    .. math::
        \\begin{bmatrix} X_0 \\\\ X_1 \\\\ X_2 \\end{bmatrix}
        = \\frac{1}{3}
        \\begin{bmatrix} 1 & 1 & 1 \\\\ 1 & a & a^2 \\\\ 1 & a^2 & a \\end{bmatrix}
        \\begin{bmatrix} X_a \\\\ X_b \\\\ X_c \\end{bmatrix}

    Zero sequence exists only when there is a ground return path, so
    :math:`\\Delta I_0` separates grounded faults (LG, LLG) from ungrounded
    ones (LL, LLL).  Negative sequence appears for any unbalanced fault.  The
    magnitude of :math:`\\Delta I_0` also decays with distance in a way that
    differs from the positive-sequence decay, because the zero-sequence path
    includes the neutral and earth return — which is precisely the extra
    information that discriminates between laterals.

    ``ratio_dI2_dI1`` and ``ratio_dI0_dI1`` normalise the unbalance out of the
    fault severity, leaving a nearly severity-independent fingerprint of the
    fault type and path.

**Apparent power** — ``Sb``, ``Sf``, ``dS``

    :math:`|V| \\cdot |I|` per phase, before, during and the difference.
    A coarse energy-flow indicator that helps separate faults upstream of a PMU
    (power reverses) from faults downstream.

**Aggregates over PMUs** — ``severity``, ``pmu_max_dI``, ``pmu_max_dV``

    Mean superimposed current across all PMUs and phases, and the index of the
    PMU that saw the largest current and voltage change.  The two ``argmax``
    features encode "which PMU is electrically closest to the fault" — a strong
    coarse localisation cue.  They are **categorical**: the value 3 is not
    larger than the value 1, it is a different PMU.  They must be one-hot
    encoded before any scaler is fitted (see
    :data:`config.CATEGORICAL_FEATURES`).

Subset dependence
-----------------
``severity``, ``pmu_max_dI`` and ``pmu_max_dV`` are defined *over the set of
PMUs given*.  When a K-PMU scenario is extracted from the full candidate
dataset, they must be recomputed on the K selected PMUs, never copied — copying
would leak information from PMUs the scenario does not have.  See
:func:`rebuild_global_features`.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

#: Fortescue rotation operator :math:`a = e^{j 2\pi/3} = 1\angle 120^\circ`.
A_OPERATOR = np.exp(1j * 2 * np.pi / 3)

#: Columns describing the simulated scenario.  They are ground truth about the
#: fault, not measurements, so they are dropped before training: a PMU cannot
#: observe the fault resistance or the irradiance.
SCENARIO_COLUMNS = (
    "impedancia_falta",
    "tipo_falta",
    "fator_carga",
    "irradiancia_global",
    "fases",
)

MEASUREMENT_PREFIXES = (
    "Vb_", "Vf_", "AngVb_", "AngVf_", "Ib_", "If_", "AngIb_", "AngIf_",
)

_RE_PMU_PHASE = re.compile(r"_(\d+)_f\d+$")
_RE_PMU_PLAIN = re.compile(r"_(\d+)$")


def phasor_delta(
    mag_fault: np.ndarray,
    ang_fault_rad: np.ndarray,
    mag_pre: np.ndarray,
    ang_pre_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Superimposed phasor :math:`\\tilde X_f - \\tilde X_b`, in polar form.

    The subtraction is done in rectangular coordinates and converted back, so
    the result is a true phasor difference.  Subtracting magnitudes and angles
    independently would be wrong: a 10° rotation at constant magnitude is a
    large superimposed phasor, but would appear as zero magnitude change.

    Parameters
    ----------
    mag_fault, mag_pre
        Magnitudes, in volts or amperes.
    ang_fault_rad, ang_pre_rad
        Angles in **radians**.

    Returns
    -------
    (magnitude, angle_rad)
        Magnitude in the same unit as the inputs; angle wrapped to (-π, π].
    """
    fx = mag_fault * np.cos(ang_fault_rad)
    fy = mag_fault * np.sin(ang_fault_rad)
    bx = mag_pre * np.cos(ang_pre_rad)
    by = mag_pre * np.sin(ang_pre_rad)
    dx, dy = fx - bx, fy - by
    return np.hypot(dx, dy), np.arctan2(dy, dx)


def sequence_components(
    xa: np.ndarray, xb: np.ndarray, xc: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-, positive- and negative-sequence components of a phasor triplet.

    Inputs and outputs are complex arrays.  Phase order is assumed ABC.
    """
    x0 = (xa + xb + xc) / 3.0
    x1 = (xa + A_OPERATOR * xb + A_OPERATOR**2 * xc) / 3.0
    x2 = (xa + A_OPERATOR**2 * xb + A_OPERATOR * xc) / 3.0
    return x0, x1, x2


def extract_pmu_id(column: str) -> str | None:
    """PMU bus identifier encoded in a feature name, or ``None`` if global.

    ``"Z_est_63_f2"`` -> ``"63"``; ``"abs_dI0_63"`` -> ``"63"``;
    ``"severity"`` -> ``None``.

    .. warning::
       Do not replace this with ``column.endswith("_1")``-style tests.  A
       one-hot column such as ``pmu_max_dI_1`` ends in ``_1`` but refers to a
       *position in the PMU list*, not to bus 1.  Mixing the two silently
       corrupts the per-PMU feature index used for missing-PMU imputation.
    """
    match = _RE_PMU_PHASE.search(column) or _RE_PMU_PLAIN.search(column)
    return match.group(1) if match else None


def build_features(
    df: pd.DataFrame,
    pmus: Sequence[str],
    eps: float = 1e-6,
    phases: Sequence[str] = ("f1", "f2", "f3"),
    verbose: bool = True,
) -> pd.DataFrame:
    """Derive the full physics-informed feature matrix.

    Parameters
    ----------
    df
        Raw simulation output, containing ``Vb_/Vf_/Ib_/If_`` magnitudes and
        ``AngVb_/AngVf_/AngIb_/AngIf_`` angles **in degrees** for every bus in
        ``pmus`` and every phase.
    pmus
        Buses to build features for.  Measurement columns belonging to other
        buses are dropped, so this doubles as the PMU subset selector.
    eps
        Guard for divisions and logarithms.
    phases
        Phase suffixes, in ABC order.
    verbose
        Print shape and sanity information.

    Returns
    -------
    pandas.DataFrame
        The raw measurement columns of the selected PMUs, plus every derived
        feature, plus the target column.  Scenario columns and DG metadata are
        removed.  Row order is preserved, which is what keeps the frozen splits
        valid.

    Raises
    ------
    ValueError
        If any expected measurement column is missing.
    """
    pmus = [str(p) for p in pmus]
    if verbose:
        print(f"  Building features for {len(pmus)} PMU(s): {pmus}")

    missing = [
        f"{prefix}{pmu}_{ph}"
        for pmu in pmus
        for ph in phases
        for prefix in MEASUREMENT_PREFIXES
        if f"{prefix}{pmu}_{ph}" not in df.columns
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} measurement column(s) missing from the raw "
            f"dataset, first few: {missing[:5]}. "
            "Were these PMUs instrumented during the simulation campaign?"
        )

    derived: dict[str, np.ndarray] = {}
    delta_v: dict[str, dict[str, np.ndarray]] = {p: {} for p in pmus}
    delta_i: dict[str, dict[str, np.ndarray]] = {p: {} for p in pmus}
    # |ΔV| and |ΔI| are kept aside to build the aggregates, then discarded:
    # they duplicate dfV_/dfI_, which are already non-negative magnitudes.
    abs_dv: dict[str, np.ndarray] = {}
    abs_di: dict[str, np.ndarray] = {}

    # ── per PMU, per phase ────────────────────────────────────────────────
    for pmu in pmus:
        for ph in phases:
            tag = f"{pmu}_{ph}"

            v_pre = df[f"Vb_{tag}"].to_numpy(dtype=float)
            v_flt = df[f"Vf_{tag}"].to_numpy(dtype=float)
            i_pre = df[f"Ib_{tag}"].to_numpy(dtype=float)
            i_flt = df[f"If_{tag}"].to_numpy(dtype=float)
            ang_v_pre = np.deg2rad(df[f"AngVb_{tag}"].to_numpy(dtype=float))
            ang_v_flt = np.deg2rad(df[f"AngVf_{tag}"].to_numpy(dtype=float))
            ang_i_pre = np.deg2rad(df[f"AngIb_{tag}"].to_numpy(dtype=float))
            ang_i_flt = np.deg2rad(df[f"AngIf_{tag}"].to_numpy(dtype=float))

            # Per-unit sag / surge.  Where the pre-fault magnitude is
            # numerically zero the ratio is undefined, so it is pinned to 1
            # ("no change") rather than left to overflow.
            ratio_v = np.where(np.abs(v_pre) > eps, v_flt / (v_pre + eps), 1.0)
            ratio_i = np.where(np.abs(i_pre) > eps, i_flt / (i_pre + eps), 1.0)

            dv_mag, dv_ang = phasor_delta(v_flt, ang_v_flt, v_pre, ang_v_pre)
            di_mag, di_ang = phasor_delta(i_flt, ang_i_flt, i_pre, ang_i_pre)

            delta_v[pmu][ph] = dv_mag * np.exp(1j * dv_ang)
            delta_i[pmu][ph] = di_mag * np.exp(1j * di_ang)
            abs_dv[tag] = dv_mag
            abs_di[tag] = di_mag

            # Superimposed apparent impedance |ΔV/ΔI| and its argument.
            z_mag = dv_mag / (di_mag + eps)
            z_ang = dv_ang - di_ang

            derived[f"ratioV_{tag}"] = ratio_v
            derived[f"ratioI_{tag}"] = ratio_i
            derived[f"logRV_{tag}"] = np.log(np.clip(np.abs(ratio_v), eps, None))
            derived[f"logRI_{tag}"] = np.log(np.clip(np.abs(ratio_i), eps, None))

            derived[f"dfV_{tag}"] = dv_mag
            derived[f"dfI_{tag}"] = di_mag
            # Angles are exported in degrees for readability, and additionally
            # as sin/cos so that the models never see the ±180° discontinuity:
            # -179° and +179° are adjacent on the circle but maximally distant
            # as raw numbers.
            derived[f"dfAngV_{tag}"] = np.rad2deg(dv_ang)
            derived[f"dfAngI_{tag}"] = np.rad2deg(di_ang)
            derived[f"sin_dfAngV_{tag}"] = np.sin(dv_ang)
            derived[f"cos_dfAngV_{tag}"] = np.cos(dv_ang)
            derived[f"sin_dfAngI_{tag}"] = np.sin(di_ang)
            derived[f"cos_dfAngI_{tag}"] = np.cos(di_ang)

            derived[f"Z_est_{tag}"] = z_mag
            derived[f"log_absZ_{tag}"] = np.log(np.clip(z_mag, eps, None))
            derived[f"sin_Z_ang_{tag}"] = np.sin(z_ang)
            derived[f"cos_Z_ang_{tag}"] = np.cos(z_ang)

            derived[f"Sb_{tag}"] = v_pre * i_pre
            derived[f"Sf_{tag}"] = v_flt * i_flt
            derived[f"dS_{tag}"] = v_flt * i_flt - v_pre * i_pre

        # Spread of |ΔI| across phases.  Near zero for a balanced three-phase
        # fault, large for a single-phase one — a direct fault-type cue that
        # also carries directional information.
        stack = np.stack([abs_di[f"{pmu}_{ph}"] for ph in phases])
        derived[f"unbalance_I_{pmu}"] = stack.max(axis=0) - stack.min(axis=0)

    # ── symmetrical components of the superimposed phasors ────────────────
    for pmu in pmus:
        dv0, dv1, dv2 = sequence_components(*(delta_v[pmu][ph] for ph in phases))
        di0, di1, di2 = sequence_components(*(delta_i[pmu][ph] for ph in phases))

        derived[f"abs_dV0_{pmu}"] = np.abs(dv0)
        derived[f"abs_dV1_{pmu}"] = np.abs(dv1)
        derived[f"abs_dV2_{pmu}"] = np.abs(dv2)
        derived[f"abs_dI0_{pmu}"] = np.abs(di0)
        derived[f"abs_dI1_{pmu}"] = np.abs(di1)
        derived[f"abs_dI2_{pmu}"] = np.abs(di2)
        derived[f"ratio_dI2_dI1_{pmu}"] = np.abs(di2) / (np.abs(di1) + eps)
        derived[f"ratio_dI0_dI1_{pmu}"] = np.abs(di0) / (np.abs(di1) + eps)

    # ── aggregates over the PMU set ───────────────────────────────────────
    n_cells = len(pmus) * len(phases)
    derived["severity"] = (
        sum(abs_di[f"{p}_{ph}"] for p in pmus for ph in phases) / n_cells
    )
    derived["pmu_max_dI"] = _argmax_over_pmus(abs_di, pmus, phases)
    derived["pmu_max_dV"] = _argmax_over_pmus(abs_dv, pmus, phases)

    # ── assemble ──────────────────────────────────────────────────────────
    drop = [c for c in SCENARIO_COLUMNS if c in df.columns]
    drop += [c for c in df.columns if c.startswith("gd")]
    drop += _measurement_columns_of_other_pmus(df.columns, set(pmus))

    out = pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1)
    out = out.drop(columns=[c for c in drop if c in out.columns])

    n_nan = int(out.isna().to_numpy().sum())
    n_inf = int(np.isinf(out.select_dtypes(include=[np.number]).to_numpy()).sum())
    if n_nan or n_inf:
        if verbose:
            print(f"  Replacing {n_nan} NaN and {n_inf} Inf value(s) with 0.")
        out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    if verbose:
        print(f"  Derived features: {len(derived)}")
        print(f"  Final shape: {out.shape}")
    return out


def _argmax_over_pmus(
    magnitudes: dict[str, np.ndarray],
    pmus: Sequence[str],
    phases: Sequence[str],
) -> np.ndarray:
    """Index of the PMU with the largest phase-summed magnitude.

    The returned value indexes ``pmus``, i.e. it is a **position**, not a bus
    number.  Categorical; see :data:`config.CATEGORICAL_FEATURES`.
    """
    totals = np.column_stack(
        [sum(magnitudes[f"{p}_{ph}"] for ph in phases) for p in pmus]
    )
    return np.argmax(totals, axis=1).astype(float)


def _measurement_columns_of_other_pmus(
    columns: Iterable[str], keep: set[str]
) -> list[str]:
    """Raw measurement columns belonging to PMUs outside ``keep``."""
    out = []
    for col in columns:
        for prefix in MEASUREMENT_PREFIXES:
            if col.startswith(prefix):
                bus = col[len(prefix):].rsplit("_", 1)[0]
                if bus not in keep:
                    out.append(col)
                break
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  SUBSET EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════
def select_pmu_subset(
    df: pd.DataFrame,
    pmus: Sequence[str],
    target_col: str,
    global_features: Iterable[str] = ("severity", "pmu_max_dI", "pmu_max_dV"),
    phases: Sequence[str] = ("f1", "f2", "f3"),
    verbose: bool = True,
) -> pd.DataFrame:
    """Extract a K-PMU feature matrix from the full candidate matrix.

    Per-PMU features are copied verbatim: every one of them is a function of
    that PMU's own phasors only, so removing other PMUs cannot change them.
    The three global features are **recomputed** from the retained ``dfI_``
    and ``dfV_`` columns, because their definition ranges over the PMU set and
    copying them would leak the behaviour of PMUs the scenario does not have.

    This is both cheaper and more auditable than re-deriving everything with
    :func:`build_features`: exactly three columns change, and the row order is
    preserved so the frozen splits stay valid.

    Raises
    ------
    ValueError
        If the ``dfI_``/``dfV_`` columns needed to rebuild the globals are
        absent for any requested PMU.
    """
    pmus = [str(p) for p in pmus]
    global_features = set(global_features)
    keep: list[str] = [target_col]
    n_dropped_global = 0

    for col in df.columns:
        if col == target_col:
            continue
        if col in global_features:
            n_dropped_global += 1
            continue
        pmu = extract_pmu_id(col)
        if pmu is None:
            n_dropped_global += 1
            continue
        if pmu in pmus:
            keep.append(col)

    subset = df[keep].copy()
    if verbose:
        print(f"    Columns kept: {len(keep)} (target included)")
        print(f"    Global columns to be recomputed: {n_dropped_global}")

    return rebuild_global_features(subset, pmus, phases=phases, verbose=verbose)


def rebuild_global_features(
    df: pd.DataFrame,
    pmus: Sequence[str],
    phases: Sequence[str] = ("f1", "f2", "f3"),
    verbose: bool = True,
) -> pd.DataFrame:
    """Recompute ``severity``, ``pmu_max_dI`` and ``pmu_max_dV`` in place.

    ``dfI_`` and ``dfV_`` are already magnitudes of superimposed phasors and
    therefore non-negative, so no absolute value is needed.
    """
    missing_i = [
        f"dfI_{p}_{ph}" for p in pmus for ph in phases
        if f"dfI_{p}_{ph}" not in df.columns
    ]
    missing_v = [
        f"dfV_{p}_{ph}" for p in pmus for ph in phases
        if f"dfV_{p}_{ph}" not in df.columns
    ]
    if missing_i or missing_v:
        raise ValueError(
            "Cannot rebuild the global features: missing "
            f"{len(missing_i)} dfI_ and {len(missing_v)} dfV_ column(s). "
            f"First few: {(missing_i + missing_v)[:5]}"
        )

    di_cols = [f"dfI_{p}_{ph}" for p in pmus for ph in phases]
    df["severity"] = df[di_cols].mean(axis=1).to_numpy()

    totals_i = np.column_stack(
        [df[[f"dfI_{p}_{ph}" for ph in phases]].sum(axis=1).to_numpy() for p in pmus]
    )
    totals_v = np.column_stack(
        [df[[f"dfV_{p}_{ph}" for ph in phases]].sum(axis=1).to_numpy() for p in pmus]
    )
    df["pmu_max_dI"] = np.argmax(totals_i, axis=1).astype(float)
    df["pmu_max_dV"] = np.argmax(totals_v, axis=1).astype(float)

    n_nan = int(df.isna().to_numpy().sum())
    n_inf = int(np.isinf(df.select_dtypes(include=[np.number]).to_numpy()).sum())
    if n_nan or n_inf:
        if verbose:
            print(f"    Replacing {n_nan} NaN and {n_inf} Inf value(s) with 0.")
        df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    if verbose:
        counts_i = dict(zip(*np.unique(df["pmu_max_dI"].astype(int), return_counts=True)))
        print(f"    severity: mean={df['severity'].mean():.4g}")
        print(f"    pmu_max_dI distribution: {counts_i}")
        print(f"    Final shape: {df.shape}")
    return df
