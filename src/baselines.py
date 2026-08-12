"""Classical fault-location baselines: impedance-based and topological.

Two reference methods that use no learning, evaluated on exactly the same test
folds as the trained models.  They exist to answer two distinct objections a
reviewer will raise.

**Impedance baseline — "is machine learning necessary at all?"**

The textbook approach compares the apparent impedance seen at a measurement
point against the accumulated series impedance of the path to each candidate
bus, and picks the closest match:

.. math::

    \\mathrm{score}(b) = \\sum_{p,\\varphi} |\\Delta I_{p\\varphi}|^2 \\,
        \\bigl| \\tilde Z^{\\text{app}}_{p\\varphi} - \\tilde Z^{\\text{path}}_{pb} \\bigr|^2

with :math:`\\tilde Z^{\\text{app}} = \\Delta\\tilde V / \\Delta\\tilde I` and
:math:`\\tilde Z^{\\text{path}} = \\sum_{\\text{lines}} (R + jX)`.  All three
phases contribute, weighted by :math:`|\\Delta I|^2`, so that a phase carrying
no fault current cannot dominate.

The relative variant divides by :math:`|\\tilde Z^{\\text{path}}|^2`.  This
matters because the fault resistance is stochastic and unknown: under the
absolute criterion a large :math:`R_f` inflates every residual, and does so
most for nearby buses whose path impedance is small, systematically biasing the
answer outwards.  The relative form measures a *fractional* impedance
discrepancy and is largely free of that bias.

This is a heuristic of electrical distance, not a literal distance relay.  Its
expected failure on this feeder is precisely the paper's motivation: with four
photovoltaic units injecting fault current from inside the network, the
assumption that a PMU sees current flowing towards the fault is violated, and
:math:`\\tilde Z^{\\text{app}}` no longer maps monotonically onto distance.

**Topological baseline — "is the model just learning the feeder map?"**

Uses no electrical magnitude at all: only which PMU saw the most superimposed
current, and how many hops each candidate bus is from it.

.. math:: \\mathrm{score}(b) = \\sum_{p} \\Bigl(\\sum_{\\varphi} |\\Delta I_{p\\varphi}|^2\\Bigr)\\, \\mathrm{hop}(p, b)

If the trained models scored no better than this, they would be recovering the
topology and nothing more.  Keeping the two baselines separate — rather than
combining them into one hybrid — is what makes that decomposition possible.

Both are irreducibly limited by closed switches: a switch has a few milliohms
of series impedance, so the buses on either side (13/152, 18/135, 97/197) are
electrically indistinguishable to any impedance-based method.

Graph conventions
-----------------
The graph used here is **not** the classification topology from
:mod:`src.graph`.  It keeps the regulator and switch nodes that the label space
excludes, because they are needed as transit nodes for the impedance sum to
follow the real electrical path.  Fictitious buses remain valid waypoints; they
are simply never candidate answers.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import networkx as nx
import numpy as np
import pandas as pd

#: OpenDSS length-unit codes, for reporting.
DSS_UNITS = {0: "none", 1: "mi", 2: "kft", 3: "km", 4: "m", 5: "ft", 6: "in", 7: "cm"}


# ═══════════════════════════════════════════════════════════════════════════
#  ELECTRICAL GRAPH
# ═══════════════════════════════════════════════════════════════════════════
def build_impedance_graph(
    dss: Any, zero_floor: float, verbose: bool = True
) -> nx.Graph:
    """Feeder graph carrying complex series impedance on every edge.

    Edge attributes: ``z`` (``|R + jX|``, ohms), ``z_complex`` (``R + jX``) and
    ``hop`` (always 1).

    ``R1`` and ``X1`` are declared in the model as ohms *per unit of length*,
    in whatever unit that line uses, so ``(R1 + jX1) * length`` is in ohms
    regardless of which unit each individual line declares.

    Transformers and voltage regulators are inserted as near-zero-impedance
    edges. They are not conductors and their series impedance is small
    relative to the lines, but omitting them would disconnect the feeder and
    make the path impedance undefined beyond every regulator.

    Conductors whose computed impedance is numerically zero — closed switches,
    mainly — are floored at ``zero_floor`` with their original angle preserved.
    Zero-cost edges break Dijkstra's ordering and make the relative score
    divide by zero.
    """
    graph = nx.Graph()
    unit_counts: dict[int, int] = {}
    n_added = n_disabled = n_floored = 0

    i = dss.lines.first()
    while i:
        name = dss.lines.name
        dss.text(f"select Line.{name}")
        if not dss.cktelement.is_enabled:
            n_disabled += 1
            i = dss.lines.next()
            continue

        bus1 = dss.lines.bus1.split(".")[0].lower()
        bus2 = dss.lines.bus2.split(".")[0].lower()
        length = max(dss.lines.length, 0.0)
        z_complex = complex(dss.lines.r1, dss.lines.x1) * length
        unit_counts[dss.lines.units] = unit_counts.get(dss.lines.units, 0) + 1

        magnitude = abs(z_complex)
        if magnitude < zero_floor:
            angle = np.angle(z_complex) if magnitude > 0 else 0.0
            z_complex = zero_floor * np.exp(1j * angle)
            magnitude = zero_floor
            n_floored += 1

        graph.add_edge(bus1, bus2, z=magnitude, z_complex=z_complex, hop=1)
        n_added += 1
        i = dss.lines.next()

    n_transformers = 0
    i = dss.transformers.first()
    while i:
        dss.text(f"select Transformer.{dss.transformers.name}")
        if dss.cktelement.is_enabled:
            buses = [b.split(".")[0].lower() for b in dss.cktelement.bus_names]
            for a, b in zip(buses[:-1], buses[1:]):
                if a != b:
                    graph.add_edge(
                        a, b,
                        z=zero_floor,
                        z_complex=complex(zero_floor, 0.0),
                        hop=1,
                    )
                    n_transformers += 1
        i = dss.transformers.next()

    if verbose:
        print(f"  lines: {n_added} added, {n_disabled} disabled, "
              f"{n_floored} floored at {zero_floor} ohm")
        print("  length units: " + ", ".join(
            f"{DSS_UNITS.get(u, '?')}={c}" for u, c in sorted(unit_counts.items())))
        print(f"  transformers/regulators as edges: {n_transformers}")
        print(f"  graph: {graph.number_of_nodes()} nodes, "
              f"{graph.number_of_edges()} edges")

    # An empty graph means the circuit never compiled. OpenDSS reports that
    # through a dialog box and a return code the text interface swallows, so
    # without this check the failure surfaces much later as an obscure
    # networkx error about the null graph.
    if graph.number_of_nodes() == 0:
        raise ValueError(
            "The impedance graph is empty: OpenDSS reported no Line objects.\n"
            "The circuit almost certainly failed to compile — check that the "
            "master file path is correct and that any error dialog OpenDSS "
            "raised was about a missing file."
        )

    if not nx.is_connected(graph):
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        raise ValueError(
            f"Impedance graph is disconnected ({len(components)} components, "
            f"sizes {[len(c) for c in components][:5]}). Path impedance would "
            "be undefined for the isolated buses."
        )
    return graph


def compute_paths(
    graph: nx.Graph, pmus: Sequence[str], candidate_buses: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulated complex impedance and hop count from each PMU to each bus.

    Returns ``(Z_path, hops)``, both shaped ``(n_buses, n_pmus)``.

    The base configuration of this feeder is radial, so there is exactly one
    electrical path between any two nodes and the Dijkstra path and the
    breadth-first path traverse the same edges.  The complex impedance is
    summed along that path; summing magnitudes instead would discard the
    R/X ratio, which is what distinguishes a long overhead run from a short
    cable section of the same magnitude.

    Raises
    ------
    ValueError
        If a PMU or a candidate bus is missing from the graph, or if any pair
        is unreachable.  Both were previously filled with a sentinel value,
        which produced finite-looking scores from meaningless data.
    """
    missing_pmus = [p for p in pmus if p not in graph]
    if missing_pmus:
        raise ValueError(f"PMU bus(es) absent from the impedance graph: {missing_pmus}")
    missing_buses = [b for b in candidate_buses if b not in graph]
    if missing_buses:
        raise ValueError(
            f"{len(missing_buses)} candidate bus(es) absent from the impedance "
            f"graph, first few: {missing_buses[:5]}"
        )

    z_path = np.zeros((len(candidate_buses), len(pmus)), dtype=np.complex128)
    hops = np.zeros((len(candidate_buses), len(pmus)), dtype=np.int32)
    row_of = {b: i for i, b in enumerate(candidate_buses)}

    for j, pmu in enumerate(pmus):
        _, paths = nx.single_source_dijkstra(graph, pmu, weight="z")
        hop_lengths = nx.single_source_shortest_path_length(graph, pmu)
        for bus in candidate_buses:
            if bus not in paths:
                raise ValueError(f"No path from PMU {pmu!r} to bus {bus!r}.")
            path = paths[bus]
            total = complex(0.0, 0.0)
            for u, v in zip(path[:-1], path[1:]):
                total += graph[u][v]["z_complex"]
            z_path[row_of[bus], j] = total
            hops[row_of[bus], j] = hop_lengths[bus]

    return z_path, hops


# ═══════════════════════════════════════════════════════════════════════════
#  OBSERVATIONS
# ═══════════════════════════════════════════════════════════════════════════
def infer_pmus_from_columns(df: pd.DataFrame, phase: str = "f1") -> list[str]:
    """Recover the PMU list from the feature columns, in order of appearance.

    Reading the placement from the data rather than from a hard-coded list is
    what keeps this stage in step with the placement search: the list changes
    whenever FSNR is re-run, and a stale constant would silently evaluate the
    baseline on a different sensor set than the models.
    """
    pattern = re.compile(rf"^Z_est_(.+)_{phase}$")
    ordered: list[str] = []
    seen: set[str] = set()
    for col in df.columns:
        match = pattern.match(col)
        if match and match.group(1) not in seen:
            seen.add(match.group(1))
            ordered.append(match.group(1))
    return ordered


def extract_observations(
    df: pd.DataFrame,
    pmus: Sequence[str],
    phases: Sequence[str],
    z_clip: float,
    dominant_phase_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the complex apparent impedance and its weights from the features.

    The feature matrix stores the apparent impedance in polar form — magnitude
    in ``Z_est_*`` and angle as ``cos_Z_ang_*`` / ``sin_Z_ang_*`` — so the
    complex value is recovered as :math:`|Z|(\\cos\\theta + j\\sin\\theta)`.
    The impedance must be complex here: the angle is what separates a genuine
    line impedance from the resistive contribution of the fault path.

    Weights are :math:`|\\Delta I|^2`. A phase carrying no fault current has an
    essentially undefined apparent impedance, and the squared superimposed
    current suppresses it smoothly instead of requiring a threshold.

    Parameters
    ----------
    dominant_phase_only
        Keep only the phase with the largest :math:`|\\Delta I|` at each PMU.
        Off by default: the full weighted sum uses the healthy phases as well,
        which carry real information about the fault type and the ground return
        path, and needs no threshold to decide what counts as "faulted".

    Returns
    -------
    (Z_app, W)
        Both shaped ``(n_samples, n_pmus, 3)``.
    """
    n = len(df)
    z_app = np.zeros((n, len(pmus), len(phases)), dtype=np.complex128)
    weights = np.zeros((n, len(pmus), len(phases)), dtype=np.float64)

    for j, pmu in enumerate(pmus):
        for k, phase in enumerate(phases):
            magnitude = df[f"Z_est_{pmu}_{phase}"].to_numpy()
            cos_a = df[f"cos_Z_ang_{pmu}_{phase}"].to_numpy()
            sin_a = df[f"sin_Z_ang_{pmu}_{phase}"].to_numpy()
            delta_i = df[f"dfI_{pmu}_{phase}"].to_numpy()

            value = magnitude * (cos_a + 1j * sin_a)
            excessive = np.abs(value) > z_clip
            value = np.where(
                excessive, z_clip * value / (np.abs(value) + 1e-12), value
            )

            z_app[:, j, k] = value
            weights[:, j, k] = delta_i**2

    if dominant_phase_only:
        keep = np.zeros_like(weights, dtype=bool)
        np.put_along_axis(
            keep, np.argmax(weights, axis=2, keepdims=True), True, axis=2
        )
        weights = np.where(keep, weights, 0.0)

    return z_app, weights


# ═══════════════════════════════════════════════════════════════════════════
#  SCORING
# ═══════════════════════════════════════════════════════════════════════════
def score_reactance(
    z_app: np.ndarray,
    weights: np.ndarray,
    z_path: np.ndarray,
    normalize_by_path: bool,
) -> np.ndarray:
    """Reactance-only criterion: compare :math:`\\Im(Z)` instead of the modulus.

    This is the textbook defence against unknown fault resistance, and the
    reason the Takagi family of estimators is built on it.  For a radial feeder
    fed from one source, a purely resistive fault path adds to the **real**
    part of the apparent impedance and leaves the imaginary part untouched:

    .. math::
        \\tilde Z^{app} = \\tilde Z^{path} + R_f
        \\;\\Longrightarrow\\;
        \\Im(\\tilde Z^{app}) = \\Im(\\tilde Z^{path}),

    so the reactance is, in principle, an :math:`R_f`-immune measure of
    distance.  Scoring on it gives the classical method its best chance and
    makes the comparison against the learned models a fair one.

    The immunity is only approximate here, and for reasons that are exactly the
    paper's subject: it assumes a single infeed and a homogeneous line, and
    both fail on this feeder.  Distributed generation makes the fault current
    at a PMU differ from the total fault current, so the residual voltage drop
    acquires a reactive component proportional to that mismatch — the classical
    "remote infeed" error.  Whatever margin this criterion recovers over the
    modulus criterion is therefore a measure of how much of the failure is due
    to fault resistance, and whatever it fails to recover is due to the
    distributed sources.

    Returns ``(n_samples, n_buses)``; **lower is better**.
    """
    x_app = z_app.imag
    x_path = z_path.imag

    weighted_x = (weights * x_app).sum(axis=2)
    weight_total = weights.sum(axis=2)
    weighted_x_sq = (weights * x_app**2).sum(axis=2)
    path_sq = x_path**2

    if normalize_by_path:
        inverse = 1.0 / (path_sq + 1e-6)
        return (weighted_x_sq @ inverse.T
                - 2.0 * (weighted_x @ (x_path * inverse).T)
                + weight_total @ (path_sq * inverse).T)

    # The observation-only term is constant across candidate buses here, so it
    # is dropped: it shifts every score equally and changes neither the argmin
    # nor the softmax.
    return (-2.0 * (weighted_x @ x_path.T)
            + weight_total @ path_sq.T)


def score_impedance(
    z_app: np.ndarray,
    weights: np.ndarray,
    z_path: np.ndarray,
    normalize_by_path: bool,
) -> np.ndarray:
    """Weighted impedance discrepancy between observation and each candidate.

    Returns ``(n_samples, n_buses)``; **lower is better**.

    Expanded as :math:`|a-b|^2 = |a|^2 - 2\\,\\mathrm{Re}(a\\bar b) + |b|^2` and
    reduced over the phase axis *before* the matrix products, so the
    ``(N, n_buses, n_pmus, 3)`` tensor is never materialised — at 75 000
    samples and 122 buses that array would be tens of gigabytes.

    In the absolute form the term :math:`\\sum_\\varphi w|Z^{app}|^2` is
    constant across candidate buses within a sample, so it is omitted: it
    shifts every score by the same amount and changes neither the argmin nor
    the softmax. In the relative form it does **not** cancel, because it is
    divided by :math:`|Z^{path}_b|^2`, which varies with the bus.
    """
    weighted_z = (weights * z_app).sum(axis=2)                  # (N, n_pmus)
    weight_total = weights.sum(axis=2)                          # (N, n_pmus)
    weighted_abs_sq = (weights * np.abs(z_app) ** 2).sum(axis=2)
    path_abs_sq = np.abs(z_path) ** 2                           # (n_buses, n_pmus)

    if normalize_by_path:
        inverse = 1.0 / (path_abs_sq + 1e-6)
        term_observation = weighted_abs_sq @ inverse.T
        term_path = weight_total @ (path_abs_sq * inverse).T
        term_cross = -2.0 * (weighted_z @ (z_path.conj() * inverse).T).real
        return term_observation + term_cross + term_path

    term_path = weight_total @ path_abs_sq.T
    term_cross = -2.0 * (weighted_z @ z_path.conj().T).real
    return term_cross + term_path


def score_topological(weights: np.ndarray, hops: np.ndarray) -> np.ndarray:
    """Hop distance weighted by how active each PMU is.  Lower is better.

    Uses no voltage, no angle and no impedance — only the topology and which
    PMU saw the most superimposed current.
    """
    return weights.sum(axis=2) @ hops.astype(np.float64).T


def scores_to_probs(
    scores: np.ndarray, calibration: str, temperature: float | None = None
) -> tuple[np.ndarray, Any]:
    """Convert scores (lower is better) into pseudo-probabilities.

    Needed only so that Top-3 and Top-5 are defined for the baselines on the
    same footing as for the models.  These are rankings expressed as
    probabilities, not calibrated confidences, and should not be read as such.

    ``"softmax"``
        :math:`p \\propto \\exp(-\\text{score}/T)`.  With ``temperature=None``
        the scale is set per run to the median across samples of the
        within-sample standard deviation of the scores, which keeps the
        distribution from collapsing onto one bus or flattening to uniform.
    ``"rank"``
        :math:`p \\propto 1/(\\text{rank}+1)`; ignores the score magnitudes
        entirely.
    """
    if calibration == "rank":
        ranks = np.argsort(np.argsort(scores, axis=1), axis=1)
        inverse = 1.0 / (ranks + 1)
        return inverse / inverse.sum(axis=1, keepdims=True), "rank"

    if calibration != "softmax":
        raise ValueError(f"Unknown calibration {calibration!r}.")

    t = (
        max(float(np.median(scores.std(axis=1))), 1e-9)
        if temperature is None
        else float(temperature)
    )
    z = -(scores - scores.min(axis=1, keepdims=True)) / t
    np.clip(z, -50, 50, out=z)
    exponent = np.exp(z)
    return exponent / exponent.sum(axis=1, keepdims=True), t


def compile_circuit(dss: Any, dss_path, verbose: bool = True) -> Any:
    """Compile and solve the OpenDSS model, verifying that it actually worked.

    ``dss.text("compile ...")`` does not raise on failure: OpenDSS pops a modal
    dialog and returns, so a missing or malformed master file leaves an empty
    circuit that every later step happily operates on.  This wraps the sequence
    with the checks that turn such a failure into an immediate, explicit error.

    Interactive dialogs are disabled first where the binding supports it. On a
    headless or scripted run a modal dialog blocks the process indefinitely,
    which is worse than an exception.

    Raises
    ------
    SystemExit
        If the file does not exist, if the circuit compiles to nothing, or if
        the base power flow does not converge.
    """
    from pathlib import Path

    dss_path = Path(dss_path)
    if not dss_path.exists():
        raise SystemExit(
            f"OpenDSS master file not found:\n  {dss_path}\n\n"
            "Set IEEE123_DSS_MASTER for this session, e.g.\n"
            '  $env:IEEE123_DSS_MASTER = "C:\\Program Files\\OpenDSS\\'
            'IEEETestCases\\123Bus\\IEEE123Master.dss"\n'
            "Note that $env: does not persist between PowerShell windows. To "
            "set it permanently:\n"
            '  [Environment]::SetEnvironmentVariable("IEEE123_DSS_MASTER", '
            '"<path>", "User")\n'
            "Alternatively, copy the 123Bus folder into network/dss/."
        )

    for attribute in ("allow_forms", "allow_editor"):
        try:
            setattr(dss, attribute, False)
        except Exception:  # noqa: BLE001 - not every binding exposes these
            pass

    dss.text("clear")
    dss.text(f"compile [{dss_path}]")

    try:
        n_buses = len(dss.circuit.buses_names)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"OpenDSS did not produce a circuit from {dss_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if n_buses == 0:
        raise SystemExit(
            f"OpenDSS compiled {dss_path} into an empty circuit. Any error "
            "dialog it displayed names the real cause; the usual one is a "
            "Redirect inside the master file pointing at a file that was not "
            "copied alongside it."
        )

    dss.text("solve")
    if not dss.solution.converged:
        raise SystemExit(
            "The base power flow did not converge, so the path impedances "
            "would be meaningless."
        )

    if verbose:
        print(f"  compiled and solved: {n_buses} buses")
    return dss