"""Selection of the candidate pool of PMU sites.

Searching all :math:`\\binom{122}{5}` placements is infeasible, and simulating
every placement separately is worse: each one would need its own 100 000-run
Monte-Carlo campaign.  This module instead builds a **candidate pool** of
:data:`config.N_CANDIDATES` buses that is simulated once, with every candidate
instrumented.  Any K-PMU scenario is then obtained by selecting columns from
that single dataset, so the placement search (``src.placement.fsnr``) never
touches OpenDSS.

Selection criteria, in order of precedence
------------------------------------------
1. **Three-phase buses only.**  The feature set relies on symmetrical
   components (:math:`\\Delta V_0, \\Delta V_1, \\Delta V_2` and their current
   counterparts) and on phase-to-phase unbalance.  These are undefined on a
   single- or two-phase lateral.  This is a hard physical constraint, not a
   preference.
2. **Real buses only.**  Regulator terminals and open-switch stubs are removed
   (see :data:`config.BUS_ALIAS`); a PMU cannot be installed on a fictitious
   node.
3. **Forced inclusions with a physical justification.**  Three groups, all
   listed with their reason in :data:`config.FORCED_CANDIDATES`:

   * the *feeder head*, so the single-PMU reference scenario is available;
   * the *points of common coupling of the distributed generators*, because a
     PV inverter changes the direction and magnitude of the fault current
     contribution seen from that point — a PMU there observes the local
     infeed directly instead of inferring it;
   * the *sectionalising switches*, because they bound the protection zones
     an operator would actually isolate.
4. **Farthest-first traversal** for the remaining slots.  Starting from the
   forced set, repeatedly add the three-phase bus whose hop distance to the
   nearest already-selected bus is largest.  This is the greedy 2-approximation
   to the k-centre problem and it spreads the pool across the laterals instead
   of clustering it near the substation, where the fault current signatures of
   distant buses are most similar.

Farthest-first fixes the *pool*, not the placement.  The actual placement is
chosen by the accuracy-driven search in :mod:`src.placement.fsnr`; the pool
only guarantees that the search has geographically diverse options.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx

from src.dss_parser import DssModel


def eligible_buses(graph: nx.Graph, model: DssModel) -> list[str]:
    """Buses that may host a PMU: present in the graph and three-phase.

    Parameters
    ----------
    graph
        Feeder topology graph from :func:`src.graph.build_graph`.
    model
        Parsed DSS model, used for the phase count of each bus.

    Raises
    ------
    ValueError
        If fewer than :data:`config.N_CANDIDATES` buses qualify.
    """
    from config import N_CANDIDATES

    phase_count = model.bus_phase_count()
    out = [b for b in graph.nodes() if phase_count.get(b, 0) >= 3]
    out.sort(key=lambda b: (int(b), "") if b.isdigit() else (10**9, b))

    if len(out) < N_CANDIDATES:
        raise ValueError(
            f"Only {len(out)} three-phase buses available, need at least "
            f"{N_CANDIDATES}. Check the DSS parser's phase detection."
        )
    return out


def farthest_first(
    graph: nx.Graph,
    pool: list[str],
    seeds: list[str],
    n_target: int,
) -> list[tuple[str, int]]:
    """Greedy farthest-first traversal over hop distance.

    Parameters
    ----------
    graph
        Feeder topology graph.
    pool
        Buses that may be selected.
    seeds
        Buses already selected; the traversal starts from these.
    n_target
        Total size of the returned set, seeds included.

    Returns
    -------
    list of (bus, separation)
        The buses added, in the order they were added, each with the hop
        distance to its nearest already-selected bus at the moment it was
        chosen.  That number is the marginal topological coverage the bus
        contributed and is worth reporting.
    """
    if not seeds:
        raise ValueError("farthest_first needs at least one seed bus.")

    selected = list(seeds)
    added: list[tuple[str, int]] = []

    # Distance from each seed / newly added bus to everything in the pool.
    nearest: dict[str, int] = {}
    for bus in selected:
        lengths = nx.single_source_shortest_path_length(graph, bus)
        for cand in pool:
            d = lengths.get(cand)
            if d is None:
                raise ValueError(
                    f"Bus {cand!r} unreachable from {bus!r}; graph is not "
                    "connected."
                )
            nearest[cand] = min(nearest.get(cand, d), d)

    while len(selected) < n_target:
        remaining = [b for b in pool if b not in selected]
        if not remaining:
            break
        best = max(remaining, key=lambda b: (nearest[b], -_num(b)))
        selected.append(best)
        added.append((best, nearest[best]))

        lengths = nx.single_source_shortest_path_length(graph, best)
        for cand in pool:
            d = lengths.get(cand)
            if d is not None:
                nearest[cand] = min(nearest[cand], d)

    return added


def _num(bus: str) -> int:
    """Numeric key for deterministic tie-breaking."""
    return int(bus) if bus.isdigit() else 10**9


def select_candidates(
    graph: nx.Graph,
    model: DssModel,
    n_target: int | None = None,
) -> dict:
    """Build the candidate pool.

    Returns
    -------
    dict
        ``candidates``
            The pool, sorted numerically.
        ``reasons``
            ``{bus: [reason, ...]}`` — the audit trail that justifies every
            entry.  Reproduced in the paper's placement section.
        ``forced``
            The subset that was included by physical requirement.
        ``coverage``
            Output of :func:`src.graph.coverage_stats` for the whole pool,
            over **all** feeder buses.
    """
    from config import FORCED_CANDIDATES, HEURISTIC_PMU_SET, N_CANDIDATES
    from src.graph import coverage_stats

    n_target = N_CANDIDATES if n_target is None else n_target
    pool = eligible_buses(graph, model)
    pool_set = set(pool)

    reasons: dict[str, list[str]] = {}
    forced: list[str] = []

    for bus, reason in FORCED_CANDIDATES.items():
        if bus not in pool_set:
            # Not an error: some sectionalising switches sit on two-phase
            # laterals and simply cannot host a three-phase PMU.
            continue
        forced.append(bus)
        reasons.setdefault(bus, []).append(reason)

    for bus in HEURISTIC_PMU_SET:
        if bus not in pool_set:
            continue
        if bus not in forced:
            forced.append(bus)
        reasons.setdefault(bus, []).append(
            "placement used by the heuristic baseline (kept for comparison)"
        )

    added = farthest_first(graph, pool, forced, n_target)
    for bus, separation in added:
        reasons.setdefault(bus, []).append(
            f"topological coverage ({separation} hops from the nearest "
            "already-selected candidate)"
        )

    candidates = sorted(forced + [b for b, _ in added], key=_num)
    coverage = coverage_stats(graph, candidates)

    return {
        "candidates": candidates,
        "reasons": reasons,
        "forced": forced,
        "coverage": coverage,
        "n_eligible": len(pool),
    }


def save_candidates(result: dict, path: str | Path) -> None:
    """Persist the candidate pool."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_candidates(path: str | Path) -> dict:
    """Load the candidate pool written by :func:`save_candidates`."""
    with open(path, "rb") as fh:
        return pickle.load(fh)
