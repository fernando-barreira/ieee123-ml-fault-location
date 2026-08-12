"""Topology graph of the IEEE 123-bus feeder and hop-distance utilities.

The graph is an undirected ``networkx.Graph`` whose nodes are the 122 physical
buses that can host a fault and whose edges are the closed conductors between
them.  Two attributes are carried on every edge:

``length``
    Conductor length in the units of the OpenDSS model (miles).
``is_switch``
    Whether the conductor is a switch rather than a line section.

What the graph is used for
--------------------------
* **Hop distance** between a fault bus and a PMU.  This is the natural error
  metric for a bus-classification fault locator: predicting an adjacent bus is
  operationally almost as useful as predicting the right one, whereas
  predicting a bus on a different lateral sends the crew to the wrong place.
* **Neighbourhood refinement** in the PMU placement search, where each selected
  PMU is swapped with buses within a few hops of it.
* **Coverage statistics** reported in the paper.

Modelling caveats worth stating explicitly in a paper
------------------------------------------------------
* Hop distance is a *topological* metric, not an electrical one.  Two buses one
  hop apart across a 0.5 mile section are far more distinguishable than two
  buses one hop apart across a switch of nominal length.  Switch-adjacent bus
  pairs (13/152, 18/135, 97/197) are electrically near-identical and account
  for a large share of the residual classification error; see
  :func:`switch_adjacent_pairs`.
* Normally-open ties are excluded, so the graph is a tree in the base
  configuration.  A reconfiguration study would need the ties restored.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx

from src.dss_parser import DssModel, parse_dss


def build_graph(dss_path: str | Path, model: DssModel | None = None) -> nx.Graph:
    """Build the feeder topology graph from the OpenDSS master file.

    Parallel conductors between the same pair of buses (rare in this model)
    are collapsed into the shorter one, since fault current preferentially
    follows the lower-impedance path.

    Parameters
    ----------
    dss_path
        Path to ``IEEE123Master.dss``.
    model
        A pre-parsed :class:`~src.dss_parser.DssModel`, to avoid parsing the
        file twice when the caller already has one.

    Returns
    -------
    networkx.Graph
        Undirected graph with ``length`` and ``is_switch`` edge attributes.
    """
    model = parse_dss(dss_path) if model is None else model

    switch_names = {
        (ln.bus1, ln.bus2) for ln in model.lines if ln.is_switch and not ln.is_open
    }

    graph = nx.Graph()
    for u, v, length in model.electrical_edges():
        is_switch = (u, v) in switch_names or (v, u) in switch_names
        if graph.has_edge(u, v):
            if length < graph[u][v]["length"]:
                graph[u][v]["length"] = length
            graph[u][v]["is_switch"] = graph[u][v]["is_switch"] and is_switch
        else:
            graph.add_edge(u, v, length=length, is_switch=is_switch)
    return graph


def validate_graph(graph: nx.Graph, expected_nodes: int | None = None) -> dict:
    """Check the structural invariants the rest of the pipeline relies on.

    A silently malformed graph is the most damaging failure mode in this
    project: it does not raise, it just produces wrong hop distances, wrong
    neighbourhoods for the placement search and wrong coverage numbers in the
    paper.  This function turns those into loud errors.

    Checks performed
    ----------------
    * The graph is **connected**.  A radial distribution feeder with all
      normally-closed switches closed must be a single connected component;
      an isolated node means an edge was dropped by the parser.
    * The graph is a **tree** (``|E| = |V| - 1``).  Warned, not raised: a mesh
      would indicate that a normally-open tie was left closed.
    * No self-loops.

    Returns
    -------
    dict
        Summary statistics, suitable for logging or for a table in the paper.
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("Empty graph: the DSS parser matched no Line objects.")

    if not nx.is_connected(graph):
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        detail = ", ".join(
            f"{len(c)} nodes ({sorted(c)[:5]}...)" for c in components[:4]
        )
        raise ValueError(
            f"Topology graph is disconnected ({len(components)} components: "
            f"{detail}). Every shortest-path distance computed on it would be "
            "undefined or infinite. Check the bus alias map and the list of "
            "normally-open switches."
        )

    if list(nx.selfloop_edges(graph)):
        raise ValueError("Topology graph contains self-loops.")

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    is_tree = n_edges == n_nodes - 1

    if expected_nodes is not None and n_nodes != expected_nodes:
        raise ValueError(
            f"Graph has {n_nodes} buses, expected {expected_nodes}. "
            "The label space of the classifier and the graph must agree."
        )

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "is_tree": is_tree,
        "n_switch_edges": sum(
            1 for _, _, d in graph.edges(data=True) if d.get("is_switch")
        ),
        "diameter_hops": nx.diameter(graph),
    }


def save_graph(graph: nx.Graph, path: str | Path) -> None:
    """Persist the graph with :mod:`pickle`.

    Only one topology file is ever written by this pipeline, at
    :data:`config.GRAPH_PKL`.  Every consumer loads that exact file, so hop
    distances reported in the paper cannot drift between scripts.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(path: str | Path) -> nx.Graph:
    """Load the topology graph written by :func:`save_graph`."""
    with open(path, "rb") as fh:
        graph = pickle.load(fh)
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"{path} does not contain a networkx.Graph.")
    return graph


def hop_distance_matrix(
    graph: nx.Graph, sources: list[str], targets: list[str]
) -> tuple["list[list[int]]", dict[str, int], dict[str, int]]:
    """Hop distances between two lists of buses.

    Distances come from :func:`networkx.single_source_shortest_path_length`,
    i.e. from breadth-first search over the **edges of the graph**.  They are
    not derived from bus numbering: bus labels in the IEEE 123 model are not
    ordered along the feeder, so ``|97 - 197|`` says nothing about how far
    apart those buses are.

    Returns
    -------
    matrix
        ``matrix[i][j]`` = hops from ``sources[i]`` to ``targets[j]``.
    src_index, tgt_index
        Maps from bus name to row / column index.
    """
    src_index = {b: i for i, b in enumerate(sources)}
    tgt_index = {b: j for j, b in enumerate(targets)}
    matrix = [[-1] * len(targets) for _ in sources]

    for bus, i in src_index.items():
        if bus not in graph:
            raise KeyError(f"Bus {bus!r} is not in the topology graph.")
        lengths = nx.single_source_shortest_path_length(graph, bus)
        for tgt, j in tgt_index.items():
            if tgt not in lengths:
                raise ValueError(
                    f"No path between {bus!r} and {tgt!r}: the graph should be "
                    "connected. Run validate_graph() first."
                )
            matrix[i][j] = lengths[tgt]
    return matrix, src_index, tgt_index


def coverage_stats(graph: nx.Graph, pmu_buses: list[str]) -> dict:
    """Hop distance from **every** feeder bus to its nearest PMU.

    The distribution is computed over all buses in the graph, three-phase and
    single-phase alike.  Restricting it to three-phase buses — as an early
    version of this analysis did — flatters the placement: the buses that are
    hardest to reach are precisely the single-phase laterals, so excluding them
    removes the tail of the distribution and inflates the reported coverage.

    Returns
    -------
    dict
        ``max``, ``mean``, ``median`` hop distance, the ``argmax`` bus, and the
        full ``histogram`` as ``{hops: n_buses}``.
    """
    if not pmu_buses:
        raise ValueError("pmu_buses is empty.")

    all_buses = list(graph.nodes())
    matrix, _, tgt_index = hop_distance_matrix(graph, pmu_buses, all_buses)

    nearest = []
    for bus in all_buses:
        j = tgt_index[bus]
        nearest.append(min(row[j] for row in matrix))

    histogram: dict[int, int] = {}
    for d in nearest:
        histogram[d] = histogram.get(d, 0) + 1

    worst = max(range(len(all_buses)), key=lambda i: nearest[i])
    ordered = sorted(nearest)
    n = len(ordered)
    median = (
        ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    )

    return {
        "n_buses": n,
        "max": max(nearest),
        "argmax_bus": all_buses[worst],
        "mean": sum(nearest) / n,
        "median": median,
        "histogram": dict(sorted(histogram.items())),
    }


def switch_adjacent_pairs(graph: nx.Graph) -> list[tuple[str, str]]:
    """Bus pairs joined by a switch.

    These pairs are the physical floor of this classification problem.  A
    closed switch has a nominal impedance of a few milliohms, so the voltage
    and current phasors measured anywhere on the feeder are, to within solver
    tolerance, identical for a fault on either side.  No amount of model
    capacity separates them; the correct reporting is to acknowledge them and
    to use hop-1 accuracy alongside exact accuracy.
    """
    return [
        (u, v) for u, v, d in graph.edges(data=True) if d.get("is_switch", False)
    ]
