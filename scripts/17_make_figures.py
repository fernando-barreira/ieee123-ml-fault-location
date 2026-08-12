"""Stage 17 — generate the paper's figures.

Three figures, each from artefacts already on disk, so a figure can never
disagree with the table it accompanies.

``hop_cdf.pdf`` (single column)
    Empirical CDF of the hop-distance error.  Reads stage 11.  The value at
    :math:`h=0` is Top-1 accuracy by construction, so this figure contains the
    headline number and shows how the remaining errors are distributed
    topologically.
    This figure was not included in the paper due to the 10-page limit, but is
    retained here to provide a clearer visualization of the hop-distance data.

``noise_curve.pdf`` (single column)
    Top-1 against measurement noise.  Reads stage 15.  Shows the
    accuracy-versus-resilience trade-off: the dense networks start far higher
    and fall faster, the tree ensembles start lower and barely move.  Where the
    curves cross is the noise level above which the ranking inverts, and it is
    computed and annotated rather than eyeballed.  Each model's own training
    noise level is marked with a star, so the slope immediately above the
    in-distribution point is readable.

``per_bus_accuracy.pdf`` (full width)
    Per-bus recall, computed from the stage-9 predictions.  Shows the ranked recall
    profile over **every** evaluated bus, indicating where the threshold is crossed
    and where the switch-equivalent pairs sit.  Ranking on the horizontal axis
    replaces per-bus tick labels, which are unreadable at print size with 122 buses,
    and a truncated view of the tail alone cannot establish bimodality—it hides the
    mode against which the tail is contrasted.

Usage
-----
.. code-block:: bash

    python scripts/17_make_figures.py
    python scripts/17_make_figures.py --only hop_cdf
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
from src.analysis import load_preprocessing, ordered_models, per_bus_recall
from src.plotting import (
    COLOR_HIGHLIGHT,
    COLOR_NEUTRAL,
    COLUMN_WIDTH,
    FULL_WIDTH,
    apply_style,
    canonical,
    save,
    style_for,
)


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — hop-distance CDF
# ═══════════════════════════════════════════════════════════════════════════
def figure_hop_cdf(analysis_dir: Path, out_dir: Path, x_max: int = 10) -> None:
    import matplotlib.pyplot as plt

    path = analysis_dir / "hop_distance" / "hop_per_fold.pkl"
    if not path.exists():
        print(f"  skipped hop_cdf: {path} not found (run stage 11)")
        return
    with open(path, "rb") as fh:
        per_fold = pickle.load(fh)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
    grid = np.arange(0, x_max + 1)

    for name in ordered_models(per_fold):
        # Pooling across folds is right here: the folds are disjoint test sets,
        # so their union is each sample counted exactly once.
        deltas = np.concatenate(per_fold[name])
        curve = [(deltas <= h).mean() for h in grid]
        style = style_for(name)
        ax.plot(grid, curve, label=canonical(name), markevery=1, **style)

    # The caption calls out h = 1 as the operational "adjacent bus" region:
    # an error of one hop still sends the crew to the right stretch of feeder.
    ax.axvline(1, color="#888888", linewidth=0.6, linestyle=(0, (3, 3)), zorder=0)
    ax.annotate("adjacent bus", xy=(1, 0.25), xytext=(3, 0),
                textcoords="offset points", fontsize=9, color="#666666")

    ax.set_xlabel(r"Hop distance $h$")
    ax.set_ylabel(r"$F_\delta(h)$")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(0, x_max + 1, 2))
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc=(0.311, 0.58), fontsize=6, ncol=2, framealpha=0.95,
              edgecolor="lightgray", handlelength=1.6,
              columnspacing=1.0, labelspacing=0.3, borderaxespad=0.4)
    fig.tight_layout()
    save(fig, out_dir / "hop_cdf.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — noise curve
# ═══════════════════════════════════════════════════════════════════════════
def crossover(sigmas, curve_a, curve_b) -> float | None:
    """Noise level at which two curves cross, by linear interpolation.

    Returned in the same units as ``sigmas``. ``None`` when the curves keep
    their order over the whole sweep.
    """
    difference = np.asarray(curve_a) - np.asarray(curve_b)
    for i in range(len(difference) - 1):
        if difference[i] == 0:
            return float(sigmas[i])
        if difference[i] * difference[i + 1] < 0:
            span = difference[i] - difference[i + 1]
            fraction = difference[i] / span
            return float(sigmas[i] + fraction * (sigmas[i + 1] - sigmas[i]))
    return None


def figure_noise_curve(analysis_dir: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    robustness = analysis_dir / "robustness"
    path = robustness / "noise_independent.pkl"
    if not path.exists():
        print(f"  skipped noise_curve: {path} not found (run stage 15)")
        return
    with open(path, "rb") as fh:
        noise = pickle.load(fh)

    training_point = {}
    train_path = robustness / "noise_at_training_sigma.pkl"
    if train_path.exists():
        with open(train_path, "rb") as fh:
            payload = pickle.load(fh)
        for name, per_fold in payload["results"].items():
            sigma = payload["sigma_train"].get(name, 0.0)
            if per_fold and sigma > 0:
                training_point[name] = (
                    sigma,
                    float(np.mean([d["top1"][0] for d in per_fold.values()])),
                )

    sigmas = list(config.NOISE_SIGMAS)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
    curves = {}

    for name in ordered_models(noise):
        per_fold = noise[name]
        if not per_fold:
            continue
        mean = [float(np.mean([d[s]["top1"][0] for d in per_fold.values()]))
                for s in sigmas]
        std = [float(np.std([d[s]["top1"][0] for d in per_fold.values()]))
               for s in sigmas]
        curves[name] = mean
        style = style_for(name)

        # The training-noise point is inserted into the curve at its own x, so
        # the line passes through it and the slope just above it is visible.
        xs, ys = list(sigmas), list(mean)
        if name in training_point:
            sigma_train, accuracy = training_point[name]
            position = int(np.searchsorted(xs, sigma_train))
            xs.insert(position, sigma_train)
            ys.insert(position, accuracy)

        x_percent = [s * 100 for s in xs]
        ax.plot(x_percent, ys, label=canonical(name), **style)
        ax.fill_between([s * 100 for s in sigmas],
                        np.array(mean) - np.array(std),
                        np.array(mean) + np.array(std),
                        color=style["color"], alpha=0.12, linewidth=0)
        if name in training_point:
            sigma_train, accuracy = training_point[name]
            ax.plot([sigma_train * 100], [accuracy], marker="", markersize=9,
                    color=style["color"], linestyle="none", zorder=5)

    ax.set_xlabel("Total vector error (%)", fontsize=10)
    ax.set_ylabel("Top-1 accuracy", fontsize=10)
    ax.set_xticks([s * 100 for s in sigmas])
    ax.set_yticks((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8))
    ax.set_xticklabels([f"{s * 100:g}" for s in sigmas], rotation=60, ha="center")
    # The compliance limit of IEEE Std C37.118 for a class-P PMU. Anchoring the
    # sweep to it is what turns the x axis from an abstract noise level into an
    # instrument specification a reader can act on.
    if max(sigmas) >= 0.01:
        ax.axvline(1.0, color="#888888", linewidth=0.6, linestyle=(0, (2, 2)),
                   zorder=0)
        ax.annotate("Standard limit", xy=(1.0, ax.get_ylim()[1]),
                    xytext=(-3, -100), textcoords="offset points",
                    ha="right", fontsize=8, color="#666666")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    handles, _ = ax.get_legend_handles_labels()
    labels = ['Deep Residual MLP', 'MLP (baseline)', 'Random Forest', 'LightGBM']
    ax.legend(handles, labels, loc="upper right")

    dense = "Deep Residual MLP"
    tree = "Random Forest"
    if dense in curves and tree in curves:
        point = crossover(sigmas, curves[dense], curves[tree])
        if point is not None:
            ax.axvline(point * 100, color="#999999", linewidth=0.6,
                       linestyle=(0, (2, 2)), zorder=0)
            ax.annotate(f"crossover\nσ ≈ {point * 100:.2f}%",
                        xy=(point * 100, ax.get_ylim()[0]),
                        xytext=(4, 25), textcoords="offset points",
                        fontsize=8, color="#555555")
            print(f"  crossover between {dense} and {tree}: "
                  f"sigma = {point * 100:.3f}%")
        else:
            print(f"  no crossover between {dense} and {tree} in the swept range")

    fig.tight_layout()
    save(fig, out_dir / "noise_curve.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — per-bus accuracy
# ═══════════════════════════════════════════════════════════════════════════
def figure_per_bus(cv_dir: Path, out_dir: Path, n_worst: int, model: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Computed here from the stored predictions rather than read from a CSV
    # written by another stage. The figure and the prose report that share this
    # statistic then agree by construction, and neither depends on the other
    # having been run.
    try:
        pre = load_preprocessing(cv_dir)
    except SystemExit as exc:
        print(f"  skipped per_bus_accuracy: {exc}")
        return

    table = per_bus_recall(cv_dir, model, pre["classes_buses"], pre["n_folds"])
    if table["support"].sum() == 0:
        print(f"  skipped per_bus_accuracy: no predictions found for {model!r}")
        return

    # Every evaluated bus is shown, not only the worst.
    table = table[table["support"] > 0].sort_values("recall", ascending=False)
    recall = table["recall"].to_numpy()
    is_switch = table["switch_partner"].notna().to_numpy()
    n_buses = len(table)

    threshold = config.PER_BUS_HEALTHY_THRESHOLD
    n_healthy = int((recall >= threshold).sum())

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.0))

    # ── the ordered profile over every bus ──────────────────────────
    rank = np.arange(1, n_buses + 1)
    ax.plot(rank, recall, color=COLOR_NEUTRAL, linewidth=1.6, zorder=2)
    ax.fill_between(rank, 0, recall, color=COLOR_NEUTRAL, alpha=0.16,
                    linewidth=0, zorder=1)
    ax.scatter(rank[is_switch], recall[is_switch], s=46,
               facecolor=COLOR_HIGHLIGHT, edgecolor="white", linewidth=0.8,
               marker="o", zorder=4)

    ax.axhline(threshold, color="#444444", linewidth=0.8,
               linestyle=(0, (4, 2)), zorder=3)
    # The vertical marker puts the count where the reader can verify it: the
    # curve crosses the threshold at exactly this rank.
    if 0 < n_healthy < n_buses:
        ax.axvline(n_healthy, color="#444444", linewidth=0.8,
                   linestyle=(0, (1, 2)), zorder=3)
        ax.annotate(f"{n_healthy} of {n_buses} buses $\\geq$ {threshold:.0%}",
                    xy=(n_healthy, threshold), xytext=(68, 3.5),
                    textcoords="offset points", ha="left", fontsize=15,
                    color="#444444")

    ax.set_xlim(0.5, n_buses + 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Buses ranked by recall", fontsize=20)
    ax.set_ylabel("Per-bus Top-1 recall", fontsize=20)
    ax.set_yticks((0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    ax.tick_params(axis="both", labelsize=13, pad=0.4)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    legend_elems = [
        Line2D([], [], linestyle="none", marker="o", markersize=7,
               markerfacecolor=COLOR_HIGHLIGHT, markeredgecolor="white",
               label="Switch-equivalent pairs"),
    ]
    ax.legend(handles=legend_elems, loc="lower left", fontsize=15,
              framealpha=0.95, edgecolor="lightgray", handletextpad=0)

    fig.tight_layout()
    save(fig, out_dir / "per_bus_accuracy.pdf")
    plt.close(fig)

    # ── the numbers the caption and the text should quote ──────────────────
    below = recall < 0.70
    middle = (recall >= 0.70) & (recall < threshold)
    # Sorted descending, so the tail is the worst; reversed so the print reads
    # worst-first. Capped, because the full list belongs in the CSV.
    worst = table.tail(min(n_worst, 10)).iloc[::-1]
    print(f"  {n_healthy} of {n_buses} buses at or above {threshold:.0%}; "
          f"{int(below.sum())} below 70%; {int(middle.sum())} in between")
    print(f"  {int(is_switch.sum())} switch-equivalent buses, "
          f"median recall {np.median(recall[is_switch]):.3f} "
          f"against {np.median(recall[~is_switch]):.3f} elsewhere")
    print("  lowest: " + ", ".join(
        f"{row.bus}({row.recall:.2f})" for row in worst.itertuples()))


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--analysis-dir", type=config.resolve_path,
                        default=config.ANALYSIS_DIR)
    parser.add_argument("--cv-dir", type=config.resolve_path, default=config.CV_DIR)
    parser.add_argument("--model", type=str, default="Deep Residual MLP",
                        help="Model whose per-bus recall is plotted.")
    parser.add_argument("--out-dir", type=config.resolve_path,
                        default=config.FIGURE_DIR)
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated subset of "
                             "hop_cdf, noise_curve, per_bus.")
    parser.add_argument("--n-worst", type=int,
                        default=config.N_WORST_BUSES_PLOTTED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_dir(args.out_dir)
    apply_style()

    wanted = {f.strip() for f in args.only.split(",") if f.strip()}
    print("=" * 70)
    print("  FIGURES")
    print("=" * 70)

    if not wanted or "hop_cdf" in wanted:
        figure_hop_cdf(args.analysis_dir, args.out_dir)
    if not wanted or "noise_curve" in wanted:
        figure_noise_curve(args.analysis_dir, args.out_dir)
    if not wanted or "per_bus" in wanted:
        figure_per_bus(args.cv_dir, args.out_dir, args.n_worst, args.model)

    print(f"\n  Figures in {args.out_dir}")


if __name__ == "__main__":
    main()