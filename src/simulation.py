"""Monte-Carlo short-circuit campaign on the IEEE 123-bus feeder.

Each simulation draws one operating point and one fault, solves the network
twice, and records the phasors seen by every candidate PMU:

.. code-block:: text

    draw irradiance, load multiplier
        -> insert PV units, solve            -> PRE-FAULT snapshot  (V~, I~)
        -> insert fault object, solve        -> FAULT snapshot      (V~, I~)
        -> store both snapshots + fault labels

Recording both snapshots — rather than only the faulted one — is what makes the
superimposed (delta) quantities available downstream:

.. math::

    \\Delta \\tilde V = \\tilde V_\\text{fault} - \\tilde V_\\text{pre}, \\qquad
    \\Delta \\tilde I = \\tilde I_\\text{fault} - \\tilde I_\\text{pre}

Superimposed quantities are the classical basis of distance protection.  By
superposition, the faulted network equals the pre-fault network plus a "pure
fault" network in which the only source is a voltage injection at the fault
point.  The delta phasors therefore see the fault path *without* the load
current, which removes most of the dependence on the operating point.  That is
why the model can be trained across a ±30 % load range and a 10-100 %
irradiance range and still generalise.

Modelling assumptions
---------------------
* **Quasi-steady-state phasors.**  Every snapshot is a converged power flow at
  fundamental frequency.  Travelling waves, CT saturation and the DC decaying
  component are outside this model; the results describe a phasor-based
  locator fed by PMU or fault-recorder measurements, not a transient-based one.
* **The fault instant is not modelled.**  The pre-fault snapshot is the steady
  state immediately before the fault, and the fault snapshot the steady state
  after it, on the same operating point.  This matches how a real PMU-based
  locator works: it compares a pre-fault buffer against the post-fault window.
* **PV units are modelled as ``PVSystem`` at unity power factor.**  In the
  OpenDSS positive-sequence-plus-unbalance power-flow they behave as
  current-limited P injections, so they reduce the current drawn from the
  substation and inject fault current locally.
* **Fault detection is assumed.**  The pipeline solves *localisation given that
  a fault occurred*; detection and classification are upstream problems.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  MEASUREMENT BRANCH
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class MeasurementBranch:
    """The single line whose current a PMU at a given bus measures.

    A PMU is a physical instrument wired to one set of current transformers on
    one branch.  Fixing that branch once and reading the same branch before and
    after the fault is what makes :math:`\\Delta \\tilde I` meaningful — if the
    branch changed between the two snapshots, the subtraction would combine
    phasors measured on two different conductors.

    Attributes
    ----------
    bus
        Bus the PMU is installed at.
    line_name
        OpenDSS ``Line`` object name (without the ``Line.`` prefix).
    terminal_offset
        Index offset into ``cktelement.currents_mag_ang`` that selects the
        terminal facing ``bus``.  ``0`` for terminal 1, ``2 * n_phases`` for
        terminal 2.
    n_phases
        Number of phases of the monitored line.
    """

    bus: str
    line_name: str
    terminal_offset: int
    n_phases: int


def _incident_lines(dss: Any, bus: str) -> list[str]:
    """Names of the ``Line`` objects connected to ``bus``."""
    dss.circuit.set_active_bus(bus)
    raw = dss.bus.line_list or []
    out = []
    for item in raw:
        if not item or item == "None":
            continue
        name = item.split(".")[-1] if "." in item else item
        if item.upper().startswith("LINE.") or "." not in item:
            out.append(name)
    return out


def _terminal_offset(dss: Any, line_name: str, bus: str) -> tuple[int, int]:
    """``(offset, n_phases)`` for the terminal of ``line_name`` facing ``bus``.

    OpenDSS returns ``currents_mag_ang`` as
    ``[|I|_t1p1, ang_t1p1, |I|_t1p2, ..., |I|_t2p1, ang_t2p1, ...]``, i.e. all
    phases of terminal 1 followed by all phases of terminal 2.  Currents are
    positive **into the element** at the terminal, so reading the terminal that
    faces the PMU bus gives the current flowing away from that bus into the
    line.
    """
    dss.text(f"select Line.{line_name}")
    n_phases = dss.cktelement.num_phases
    bus1 = dss.cktelement.bus_names[0].split(".")[0].lower()
    offset = 0 if bus1 == bus.lower() else 2 * n_phases
    return offset, n_phases


def _read_terminal(dss: Any, branch: MeasurementBranch) -> list[float]:
    """Interleaved ``[|I|_A, ang_A, |I|_B, ang_B, |I|_C, ang_C]`` in A / degrees."""
    dss.text(f"select Line.{branch.line_name}")
    currents = dss.cktelement.currents_mag_ang
    start = branch.terminal_offset
    stop = start + 2 * branch.n_phases
    return list(currents[start:stop])


def _largest_current_branch(dss: Any, bus: str) -> MeasurementBranch | None:
    """Incident line carrying the largest total current magnitude.

    On a radial feeder the upstream branch carries the sum of all downstream
    load current, so it is almost always the largest.  Used only as the
    fallback for the feeder head, which has no upstream line inside the graph.
    """
    best: MeasurementBranch | None = None
    best_total = -1.0

    for prefer_three_phase in (True, False):
        for name in _incident_lines(dss, bus):
            offset, n_phases = _terminal_offset(dss, name, bus)
            if prefer_three_phase and n_phases < 3:
                continue
            branch = MeasurementBranch(bus, name, offset, n_phases)
            mags = _read_terminal(dss, branch)[0::2]
            total = float(sum(mags))
            if total > best_total:
                best_total, best = total, branch
        if best is not None:
            return best
    return best


def _upstream_branch(
    dss: Any, bus: str, hops_to_source: dict[str, int]
) -> MeasurementBranch | None:
    """Incident line whose far end is closer to the source.

    This is the electrically correct choice for a feeder-mounted PMU: it
    measures the current flowing from the source towards the bus, which is the
    quantity a protection engineer would instrument.  On a radial feeder there
    is exactly one such branch per bus (except at the source itself).
    """
    own = hops_to_source.get(bus)
    if own is None:
        return None

    best: MeasurementBranch | None = None
    best_hops = own

    for name in _incident_lines(dss, bus):
        dss.text(f"select Line.{name}")
        terminals = [b.split(".")[0].lower() for b in dss.cktelement.bus_names]
        far = [b for b in terminals if b != bus.lower()]
        if not far:
            continue
        far_hops = hops_to_source.get(far[0])
        if far_hops is None or far_hops >= best_hops:
            continue
        offset, n_phases = _terminal_offset(dss, name, bus)
        best_hops = far_hops
        best = MeasurementBranch(bus, name, offset, n_phases)
    return best


def build_branch_map(
    dss: Any,
    pmu_buses: Sequence[str],
    hops_to_source: dict[str, int],
) -> dict[str, MeasurementBranch]:
    """Decide, once, which branch each PMU monitors.

    Must be called on a **solved, unfaulted** circuit, before the Monte-Carlo
    loop starts.  The rule is fixed: each PMU monitors its **upstream** line —
    the incident conductor whose far end is closer to the source — because that
    is the branch a feeder-mounted PMU or relay would actually instrument.  The
    only bus without an upstream line is the feeder head itself, which falls
    back to the incident line carrying the largest total current in the nominal
    solve (in practice, the tie to the source).

    Freezing the branch here, and reading the same branch before and during the
    fault, is what makes the superimposed current
    :math:`\\Delta\\tilde I = \\tilde I_\\text{fault} - \\tilde I_\\text{pre}`
    physically meaningful; re-selecting per measurement could subtract phasors
    taken on two different conductors.

    Raises
    ------
    ValueError
        If a PMU bus has no usable incident line.
    """
    branch_map: dict[str, MeasurementBranch] = {}
    for bus in pmu_buses:
        branch = _upstream_branch(dss, bus, hops_to_source)
        if branch is None:
            branch = _largest_current_branch(dss, bus)
        if branch is None:
            raise ValueError(f"No line incident to PMU bus {bus!r}.")
        branch_map[bus] = branch
    return branch_map


def read_bus_current(
    dss: Any, bus: str, branch_map: dict[str, MeasurementBranch]
) -> list[float]:
    """Current phasors at a PMU, on the branch fixed by :func:`build_branch_map`.

    ``bus`` must be present in ``branch_map``; a :class:`KeyError` here is a
    programming error (the map is built for exactly the validated PMU set), not
    a data condition to be silently zero-filled.
    """
    return _read_terminal(dss, branch_map[bus])


# ═══════════════════════════════════════════════════════════════════════════
#  DISTRIBUTED GENERATION
# ═══════════════════════════════════════════════════════════════════════════
def _kv_for_phases(n_phases: int, vbase_ll: float) -> float:
    """Rated voltage of a PV unit, following the OpenDSS convention.

    ``kV`` is line-to-line for a three-phase element and line-to-neutral for a
    single-phase one, hence the :math:`\\sqrt{3}` division.  Getting this wrong
    silently scales the injected current by 1.73.
    """
    if n_phases == 1:
        return round(vbase_ll / math.sqrt(3), 4)
    return vbase_ll


def insert_pv(
    dss: Any, name: str, bus: str, kva: float, irradiance: float, vbase_ll: float
) -> int:
    """Add a ``PVSystem`` at ``bus`` matching the phases already present there.

    ``Pmpp`` is set equal to the inverter rating (unity DC/AC ratio) and
    ``pf=1``, so the unit injects active power only.  Output scales linearly
    with ``irradiance`` (per unit of 1 kW/m²), which is how the load-flow model
    represents varying insolation.

    Returns
    -------
    int
        Number of phases the unit was connected to.
    """
    dss.circuit.set_active_bus(bus)
    n_phases = dss.bus.num_nodes
    nodes = dss.bus.nodes
    kv = _kv_for_phases(n_phases, vbase_ll)
    connection = f"{bus}." + ".".join(str(n) for n in nodes)
    dss.text(
        f"New PVSystem.{name} bus1={connection} phases={n_phases} "
        f"kV={kv} kVA={kva} Pmpp={kva} pf=1 "
        f"irradiance={irradiance} %Pmpp=100"
    )
    return n_phases


# ═══════════════════════════════════════════════════════════════════════════
#  FAULT PLACEMENT
# ═══════════════════════════════════════════════════════════════════════════
def buses_by_phase_count(
    dss: Any, fictitious: Iterable[str]
) -> dict[int, list[str]]:
    """Group the solved circuit's buses by how many phases they have.

    A fault type can only be applied where the phases it needs exist: a
    three-phase fault requires a three-phase bus, a phase-to-phase fault at
    least two phases, a line-to-ground fault any bus.
    """
    fictitious = {b.lower() for b in fictitious}
    names = sorted({n.split(".")[0] for n in dss.circuit.nodes_names})

    grouped: dict[int, list[str]] = {1: [], 2: [], 3: []}
    for bus in names:
        if bus.lower() in fictitious:
            continue
        dss.circuit.set_active_bus(bus)
        n = dss.bus.num_nodes
        if n in grouped:
            grouped[n].append(bus)
    return grouped


def draw_fault_resistance(
    fault_type: str, spec: dict[str, tuple[float, float, float]], rng: np.random.RandomState
) -> float:
    """Sample a fault resistance in ohms from the log-normal model.

    Fault resistance is the single largest source of difficulty in this
    problem.  A bolted fault (``Rf -> 0``) produces a large, unambiguous
    superimposed current; a high-resistance ground fault produces a current
    barely above load variation, and the accuracy of any phasor-based locator
    degrades sharply above roughly 50 Ω.  A log-normal distribution is used
    because ``Rf`` is strictly positive and empirically right-skewed.
    """
    mu, sigma, cap = spec[fault_type]
    return float(min(rng.lognormal(mean=mu, sigma=sigma), cap))


def apply_fault(
    dss: Any, bus: str, fault_type: str, phases: Sequence[str], rf: float
) -> None:
    """Insert a ``Fault`` object implementing ``fault_type`` at ``bus``.

    ============  =========================================================
    ``fault_type``  OpenDSS realisation
    ============  =========================================================
    ``LG``          1-phase fault from the faulted phase to ground.
    ``LLG``         2-phase fault; both phases tied to ground through ``Rf``.
    ``LL``          1-phase fault between two phases, ``bus2`` on the second
                    phase, so there is **no ground return path**.
    ``LLL``         3-phase fault wired ``1.2.3`` to ``2.3.1``, i.e. a delta
                    connection between the phases and no ground return.
    ============  =========================================================

    ``r`` is the resistance of each individual fault branch, so the resistance
    seen phase-to-phase in an ``LL`` fault is ``Rf``, while an ``LLG`` fault
    has ``Rf`` from each phase to ground.

    Parameters
    ----------
    phases
        Phase node numbers as strings, e.g. ``["1", "3"]`` for an A-C fault.
    """
    if fault_type == "LG":
        node = phases[0]
        dss.text(f"New Fault.F1 phases=1 bus1={bus}.{node} r={rf}")
    elif fault_type == "LLG":
        node_list = ".".join(phases)
        dss.text(f"New Fault.F1 phases=2 bus1={bus}.{node_list} r={rf}")
    elif fault_type == "LL":
        a, b = phases[0], phases[1]
        dss.text(f"New Fault.F1 phases=1 bus1={bus}.{a} bus2={bus}.{b} r={rf}")
    elif fault_type == "LLL":
        dss.text(f"New Fault.F1 phases=3 bus1={bus}.1.2.3 bus2={bus}.2.3.1 r={rf}")
    else:
        raise ValueError(f"Unknown fault type {fault_type!r}.")


def choose_fault_phases(
    fault_type: str, available: Sequence[str], rng: random.Random
) -> list[str]:
    """Pick which phases are involved, given what the bus offers.

    Returns them in ascending node order so the recorded ``phases`` label is
    canonical (``"1.2"``, never ``"2.1"``).  The ordering has no effect on the
    physics: a phase-to-phase fault is a resistance between two nodes and is
    symmetric, and a double-line-to-ground fault grounds both listed phases.

    Raises
    ------
    ValueError
        If a two-phase fault type is requested at a bus with fewer than two
        phases.  The caller draws the fault bus from a pool filtered by phase
        count, so this indicates the pool was built incorrectly — silently
        degrading an ``LL`` fault to a single-phase one would mislabel the
        sample.
    """
    if fault_type == "LLL":
        return ["1", "2", "3"]
    if fault_type == "LG":
        return [rng.choice(list(available))]
    if len(available) < 2:
        raise ValueError(
            f"Fault type {fault_type!r} needs two phases, but the bus offers "
            f"{available}."
        )
    return sorted(rng.sample(list(available), 2), key=int)


# ═══════════════════════════════════════════════════════════════════════════
#  RECORD ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════
def record_pmu_measurements(
    row: dict,
    bus: str,
    v_pre: Sequence[float] | None,
    v_fault: Sequence[float] | None,
    i_pre: Sequence[float] | None,
    i_fault: Sequence[float] | None,
    n_phases: int = 3,
) -> None:
    """Write one PMU's phasors into ``row`` under the canonical column names.

    Column naming (``b`` = before / pre-fault, ``f`` = fault):

    =====================  ==============================================
    ``Vb_<bus>_f<n>``        pre-fault phase-``n`` voltage magnitude [V]
    ``AngVb_<bus>_f<n>``     pre-fault phase-``n`` voltage angle [deg]
    ``Vf_<bus>_f<n>``        fault voltage magnitude [V]
    ``AngVf_<bus>_f<n>``     fault voltage angle [deg]
    ``Ib_/AngIb_/If_/AngIf_``  same, for the monitored branch current [A, deg]
    =====================  ==============================================

    Missing phases are written as zeros.  All PMU sites are three-phase by
    construction, so a zero here means the solver returned fewer terminals than
    expected and should be investigated, not ignored.
    """
    for phase in range(1, n_phases + 1):
        j = (phase - 1) * 2
        tag = f"{bus}_f{phase}"

        if v_pre is not None and v_fault is not None and j + 1 < len(v_pre) and j + 1 < len(v_fault):
            row[f"Vb_{tag}"] = v_pre[j]
            row[f"AngVb_{tag}"] = v_pre[j + 1]
            row[f"Vf_{tag}"] = v_fault[j]
            row[f"AngVf_{tag}"] = v_fault[j + 1]
        else:
            row[f"Vb_{tag}"] = row[f"AngVb_{tag}"] = 0.0
            row[f"Vf_{tag}"] = row[f"AngVf_{tag}"] = 0.0

        if i_pre is not None and i_fault is not None and j + 1 < len(i_pre) and j + 1 < len(i_fault):
            row[f"Ib_{tag}"] = i_pre[j]
            row[f"AngIb_{tag}"] = i_pre[j + 1]
            row[f"If_{tag}"] = i_fault[j]
            row[f"AngIf_{tag}"] = i_fault[j + 1]
        else:
            row[f"Ib_{tag}"] = row[f"AngIb_{tag}"] = 0.0
            row[f"If_{tag}"] = row[f"AngIf_{tag}"] = 0.0
