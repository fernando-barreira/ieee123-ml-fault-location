"""Stage 2 — choose the candidate pool of PMU sites.

Selects :data:`config.N_CANDIDATES` three-phase buses that the Monte-Carlo
campaign will instrument.  Every K-PMU scenario studied later is a subset of
this pool, so the pool is simulated once and no re-simulation is needed when
the placement changes.

The selection rationale, and the audit trail written to
:data:`config.CANDIDATES_PKL`, are documented in :mod:`src.pmu_candidates`.

Usage
-----
.. code-block:: bash

    python scripts/02_select_candidates.py
    python scripts/02_select_candidates.py --n 30 --plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.dss_parser import parse_dss
from src.graph import coverage_stats, load_graph
from src.pmu_candidates import save_candidates, select_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dss", type=config.resolve_path, default=config.DSS_MASTER)
    parser.add_argument("--graph", type=config.resolve_path, default=config.GRAPH_PKL)
    parser.add_argument("--out", type=config.resolve_path, default=config.CANDIDATES_PKL)
    parser.add_argument(
        "--n", type=int, default=config.N_CANDIDATES, help="Pool size."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Draw the feeder with the pool highlighted (needs matplotlib).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_parent(args.out)

    graph = load_graph(args.graph)
    model = parse_dss(args.dss)

    result = select_candidates(graph, model, n_target=args.n)
    candidates = result["candidates"]

    print(f"Eligible three-phase buses : {result['n_eligible']}")
    print(f"Forced by physical criteria: {len(result['forced'])}")
    print(f"\nCandidate pool ({len(candidates)}):")
    for bus in candidates:
        print(f"  {bus:>5s}  {'; '.join(result['reasons'][bus])}")

    cov = result["coverage"]
    print("\nHop distance from every feeder bus to its nearest candidate")
    print("(computed over ALL buses, single-phase laterals included --")
    print(" restricting it to three-phase buses removes the hardest cases)")
    print(f"  buses considered : {cov['n_buses']}")
    print(f"  maximum          : {cov['max']} hops (bus {cov['argmax_bus']})")
    print(f"  mean             : {cov['mean']:.2f} hops")
    print(f"  median           : {cov['median']:.0f} hops")
    for hops, n in cov["histogram"].items():
        print(f"    {hops} hop(s): {n:>3d} buses")

    print("\nReference: coverage of the heuristic 5-PMU placement "
          f"{config.HEURISTIC_PMU_SET}")
    heuristic = coverage_stats(
        graph, [b for b in config.HEURISTIC_PMU_SET if b in graph]
    )
    print(
        f"  max={heuristic['max']} hops (bus {heuristic['argmax_bus']}), "
        f"mean={heuristic['mean']:.2f}"
    )

    save_candidates(result, args.out)
    print(f"\nSaved: {args.out}")

    if args.plot:
        _plot(graph, result)


def _plot(graph, result: dict) -> None:
    """Feeder layout with the candidate pool highlighted.

    The layout is a spring embedding, so distances on the page are not
    geographic; it is a qualitative check that the pool is spread over the
    laterals rather than clustered near the substation.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.patches import Patch

    forced = set(result["forced"])
    candidates = set(result["candidates"])

    colours, sizes = [], []
    for bus in graph.nodes():
        if bus == config.SUBSTATION_BUS:
            colours.append("#5b2c8d"); sizes.append(320)
        elif bus in {b for _, b, _ in config.DG_UNITS}:
            colours.append("#e07b39"); sizes.append(280)
        elif bus in forced:
            colours.append("#c0392b"); sizes.append(260)
        elif bus in candidates:
            colours.append("#2e8b57"); sizes.append(200)
        else:
            colours.append("#cfcfcf"); sizes.append(45)

    pos = nx.spring_layout(graph, seed=config.SEED, k=0.8, iterations=100)
    fig, ax = plt.subplots(figsize=(15, 9))
    nx.draw_networkx_edges(graph, pos, alpha=0.35, ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_color=colours, node_size=sizes, ax=ax)
    nx.draw_networkx_labels(
        graph, pos, labels={b: b for b in candidates}, font_size=8, ax=ax
    )
    ax.legend(
        handles=[
            Patch(color="#5b2c8d", label=f"feeder head ({config.SUBSTATION_BUS})"),
            Patch(color="#e07b39", label="PV point of common coupling"),
            Patch(color="#c0392b", label="other forced candidate"),
            Patch(color="#2e8b57", label="added for topological coverage"),
            Patch(color="#cfcfcf", label="remaining buses"),
        ],
        loc="upper left",
        fontsize=10,
    )
    ax.set_title(f"IEEE 123-bus feeder — {len(candidates)} PMU candidate sites")
    ax.axis("off")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
