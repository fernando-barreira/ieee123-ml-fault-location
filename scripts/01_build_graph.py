"""Stage 1 — build the topology graph of the IEEE 123-bus feeder.

Parses ``IEEE123Master.dss``, builds the undirected graph of closed conductors,
validates it and writes it to :data:`config.GRAPH_PKL`.

Every downstream stage loads that single file, so hop distances reported in the
paper cannot disagree between scripts.

Usage
-----
.. code-block:: bash

    python scripts/01_build_graph.py
    python scripts/01_build_graph.py --dss /path/to/IEEE123Master.dss
    python scripts/01_build_graph.py --report   # extra topology diagnostics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.dss_parser import parse_dss
from src.graph import (
    build_graph,
    save_graph,
    switch_adjacent_pairs,
    validate_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--dss",
        type=config.resolve_path,
        default=config.DSS_MASTER,
        help="Path to IEEE123Master.dss.",
    )
    parser.add_argument(
        "--out",
        type=config.resolve_path,
        default=config.GRAPH_PKL,
        help="Where to write the graph.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print switch pairs and per-bus phase counts.",
    )
    parser.add_argument(
        "--expect-buses",
        type=int,
        default=config.EXPECTED_N_CLASSES,
        help="Fail if the graph has a different number of buses. "
             "Pass 0 to skip the check (useful for reduced test networks).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_parent(args.out)

    print(f"Parsing {args.dss}")
    model = parse_dss(args.dss)
    print(f"  Line objects parsed : {len(model.lines)}")
    if model.unparsed:
        print(
            f"  WARNING: {len(model.unparsed)} Line object(s) had no Bus1/Bus2 "
            f"and were skipped: {model.unparsed[:5]}"
        )
    print(f"  Open switches       : "
          f"{sorted(l.name for l in model.lines if l.is_open)}")

    graph = build_graph(args.dss, model=model)
    stats = validate_graph(
        graph, expected_nodes=args.expect_buses or None
    )

    print("\nGraph")
    print(f"  Buses                : {stats['n_nodes']}")
    print(f"  Conductors           : {stats['n_edges']}")
    print(f"  Switch conductors    : {stats['n_switch_edges']}")
    print(f"  Radial (tree)        : {stats['is_tree']}")
    print(f"  Diameter             : {stats['diameter_hops']} hops")
    if not stats["is_tree"]:
        print(
            "  WARNING: the graph contains a loop. A distribution feeder in "
            "its base configuration should be radial; check that every "
            "normally-open tie is excluded."
        )

    three_phase = model.three_phase_buses()
    in_graph = [b for b in three_phase if b in graph]
    print(f"  Three-phase buses    : {len(in_graph)} of {stats['n_nodes']}")

    if args.report:
        print("\nSwitch-adjacent bus pairs (electrically near-identical):")
        for u, v in sorted(switch_adjacent_pairs(graph)):
            print(f"  {u:>5s} -- {v}")

        print("\nBuses by phase count:")
        counts = model.bus_phase_count()
        for n in (1, 2, 3):
            members = sorted(
                (b for b in graph if counts.get(b, 0) == n),
                key=lambda b: int(b) if b.isdigit() else 10**9,
            )
            print(f"  {n}-phase: {len(members):>3d}  {members}")

    save_graph(graph, args.out)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
