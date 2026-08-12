"""Training-length statistics for the neural models, from the stored histories.

Reports, per model and across folds, two epoch counts that are easy to conflate
and that differ by roughly the patience:

**best epoch** — where validation top-1 peaked.  The saved weights come from
here, so this is the length of training that actually produced the reported
model.

**stopping epoch** — where the run halted.  Early stopping waits ``patience``
epochs after the peak before giving up, so this exceeds the best epoch by up to
that margin and describes compute spent, not model quality.

For a sentence in a paper the best epoch is usually the honest one: saying a
model "trains for 1200 epochs" when its weights are from epoch 1100 and the
last 100 were the stopping criterion overstates what the training needed.
Quote whichever, but quote the same one consistently, and say which.

The tool also flags any fold that ran to the epoch cap, since a model that
never triggered early stopping was budget-limited rather than converged and its
result is a lower bound.

Usage
-----
.. code-block:: bash

    python tools/training_epochs.py
    python tools/training_epochs.py --cv-dir results/cv
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config


def curve_of(entry: dict) -> list[float] | None:
    """Validation-accuracy curve stored for a model, if it has one.

    Tree ensembles have no epoch structure and store ``history=None``; TabNet
    stores a single ``val_acc`` list.
    """
    history = entry.get("history")
    if not history:
        return None
    for key in ("val_acc", "val_accuracy", "valid_accuracy"):
        curve = history.get(key)
        if curve:
            return list(curve)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cv-dir", type=config.resolve_path, default=config.CV_DIR)
    parser.add_argument("--out-dir", type=config.resolve_path,
                        default=config.ANALYSIS_DIR / "training_epochs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_dir(args.out_dir)

    per_model: dict[str, list[dict]] = {}
    for fold in range(config.CV_N_SPLITS):
        path = Path(args.cv_dir) / f"fold_{fold}" / "fold_results.pkl"
        if not path.exists():
            continue
        with open(path, "rb") as fh:
            results = pickle.load(fh)
        for name, entry in results.items():
            curve = curve_of(entry)
            if curve is None:
                continue
            best = int(np.argmax(curve)) + 1
            per_model.setdefault(name, []).append({
                "fold": fold,
                "best_epoch": best,
                "stopping_epoch": len(curve),
                "best_val_acc": float(np.max(curve)),
                "training_time_s": float(entry.get("training_time", float("nan"))),
            })

    if not per_model:
        raise SystemExit(
            f"No training histories found under {args.cv_dir}. Run stage 9 "
            "first; tree ensembles do not store one."
        )

    print("=" * 78)
    print("  TRAINING LENGTH (neural models)")
    print("=" * 78)
    print(f"  epoch cap {config.CV_EPOCHS} | patience {config.CV_PATIENCE}")

    rows = []
    for name, records in per_model.items():
        best = np.array([r["best_epoch"] for r in records])
        stop = np.array([r["stopping_epoch"] for r in records])
        capped = int((stop >= config.CV_EPOCHS).sum())

        print(f"\n  {name}  ({len(records)} folds)")
        print(f"    best epoch      : {best.min()}-{best.max()}  "
              f"(mean {best.mean():.0f} +/- {best.std():.0f})")
        print(f"    stopping epoch  : {stop.min()}-{stop.max()}  "
              f"(mean {stop.mean():.0f} +/- {stop.std():.0f})")
        print(f"    per fold        : " + ", ".join(
            f"f{r['fold']}={r['best_epoch']}/{r['stopping_epoch']}"
            for r in records))
        if capped:
            print(f"    WARNING: {capped} fold(s) reached the {config.CV_EPOCHS}-epoch "
                  "cap without early stopping; those runs were budget-limited "
                  "and their accuracy is a lower bound.")

        rows += [{"model": name, **r} for r in records]

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_dir / "training_epochs.csv", index=False)

    all_best = frame["best_epoch"]
    all_stop = frame["stopping_epoch"]
    print("\n" + "=" * 78)
    print("  Across every neural model and fold")
    print("=" * 78)
    print(f"    best epoch     : {all_best.min()}-{all_best.max()}")
    print(f"    stopping epoch : {all_stop.min()}-{all_stop.max()}")

    def band(values) -> tuple[int, int]:
        """Round outward to a readable band."""
        lo = int(np.floor(values.min() / 50) * 50)
        hi = int(np.ceil(values.max() / 50) * 50)
        return lo, hi

    lo_b, hi_b = band(all_best)
    lo_s, hi_s = band(all_stop)
    print("\n  Sentences for the paper (pick one convention and keep it):")
    print(f'    "the neural models reach their best validation accuracy between '
          f'~{lo_b} and ~{hi_b} epochs"')
    print(f'    "training stops between ~{lo_s} and ~{hi_s} epochs '
          f'(patience {config.CV_PATIENCE})"')
    print(f"\n  Saved: {args.out_dir / 'training_epochs.csv'}")


if __name__ == "__main__":
    main()