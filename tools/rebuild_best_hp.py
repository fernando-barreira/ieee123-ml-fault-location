"""Rebuild ``best_hp.json`` from the per-model files written by stage 8.

Stage 8 writes ``<model>/best_params.json`` as soon as each study finishes, and
only then merges everything into ``best_hp.json``.  If that final write fails —
a read-only file, an editor holding a lock, a sync client — no search results
are lost, but stage 9 has nothing to read.  This rebuilds the consolidated file
from what is already on disk.

It also works across folders, which is what you want after re-running some
models with different search settings: point ``--from`` at each folder in turn
and the newest entry for each model wins.

Usage
-----
.. code-block:: bash

    python tools/rebuild_best_hp.py
    python tools/rebuild_best_hp.py --from results/optuna results/optuna_pruned
    python tools/rebuild_best_hp.py --out results/optuna/best_hp.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config


def collect(folder: Path) -> dict[str, dict]:
    """Read every ``<model>/best_params.json`` under ``folder``."""
    found: dict[str, dict] = {}
    if not folder.exists():
        print(f"  {folder} does not exist; skipped")
        return found

    for model_dir in sorted(folder.iterdir()):
        params_file = model_dir / "best_params.json"
        if not (model_dir.is_dir() and params_file.exists()):
            continue
        with open(params_file) as fh:
            record = json.load(fh)
        name = record.get("model", model_dir.name)
        found[name] = record
        print(f"  {folder.name}/{model_dir.name}: "
              f"top-1 {record.get('best_value', float('nan')):.4f}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--from", dest="sources", nargs="+", type=config.resolve_path,
        default=[config.OPTUNA_DIR],
        help="Folders to scan, in increasing order of precedence.",
    )
    parser.add_argument("--out", type=config.resolve_path, default=config.BEST_HP_JSON)
    args = parser.parse_args()

    merged: dict[str, dict] = {}
    for folder in args.sources:
        print(f"\nScanning {folder}")
        merged.update(collect(folder))

    if not merged:
        raise SystemExit("No best_params.json found; nothing to rebuild.")

    best_hp = {name: record["best_params"] for name, record in merged.items()}
    config.ensure_writable_file(args.out)
    with open(args.out, "w") as fh:
        json.dump(best_hp, fh, indent=2)

    print(f"\nWrote {args.out} with {len(best_hp)} model(s): {sorted(best_hp)}")
    missing = [m for m in config.MODELS if m not in best_hp]
    if missing:
        print(f"WARNING: no entry for {missing}. Stage 9 will refuse to start "
              "unless those models are passed to --skip.")


if __name__ == "__main__":
    main()