"""Shared figure style for the paper.

Two constraints drive every choice here, and both come from the venue rather
than from taste.

**Typography must match the manuscript.**  IEEEtran sets text in a Times-family
face; a figure in the matplotlib default sans-serif is immediately visible as
foreign. Times New Roman is used when installed, falling back to STIXGeneral,
which is metric-compatible and ships with matplotlib. Fonts are embedded as
Type 42 so the PDF survives the publisher's pipeline.

**Figures must survive greyscale.** Reviewers print. Colour alone therefore
carries no information: every series is distinguished three times over — by
colour, by line style and by marker — so the figures remain readable with the
colour channel discarded entirely.

Sizes are absolute, in inches, matching the IEEEtran text block: 3.5 in for a
single column and 7.0 in for the full width. Figures are never rescaled in
LaTeX, so the point sizes set here are the point sizes that appear on the page.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

#: Width of one IEEEtran column and of the full text block, in inches.
COLUMN_WIDTH = 3.5
FULL_WIDTH = 7.0

#: Alternative spellings that appear as keys in stored results.
_ALIASES = {
    "Baseline Impedance": "Impedance baseline",
    "Baseline Topological": "Topological baseline",
    "Impedance Baseline": "Impedance baseline",
    "Topological Baseline": "Topological baseline",
    "Topology baseline": "Topological baseline",
}


def canonical(name: str) -> str:
    """Resolve a model name to the spelling used in the figures."""
    return _ALIASES.get(name, name)


#: One entry per series.  Line styles are given as dash patterns where the four
#: named styles run out, so seven series remain distinguishable in greyscale.
STYLE = {
    "Deep Residual MLP":    dict(color="#1f3a5f", linestyle="-", marker="o"),
    "MLP Baseline":         dict(color="#5b8bcf", linestyle="--", marker="s"),
    "TabNet":               dict(color="#7a4f9c", linestyle="-.", marker="^"),
    "Random Forest":        dict(color="#1b7837", linestyle=":", marker="D"),
    "LightGBM":             dict(color="#7fbf7b", linestyle=(0, (3, 1, 1, 1)), marker="x"),
    "Impedance baseline":   dict(color="#c75a3a", linestyle=(0, (1, 1)), marker="P"),
    "Topological baseline": dict(color="#e0892b", linestyle=(0, (5, 2)), marker="*"),
}

_FALLBACK = dict(color="#777777", linestyle="-", marker=".")

#: Neutral fills for bar charts.  The highlight colour is the same one used for
#: the impedance baseline, so "this is the problematic case" reads consistently
#: across figures.
COLOR_NEUTRAL = "#5b7a99"
COLOR_HIGHLIGHT = "#c75a3a"

MARKERSIZE = 4.2


def style_for(name: str) -> dict:
    """Colour, line style and marker for a series."""
    return STYLE.get(canonical(name), _FALLBACK)


def apply_style() -> None:
    """Install the shared rcParams.  Call once before creating any figure."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "lines.markersize": MARKERSIZE,
        # Open markers such as 'x', '+' and '*' are invisible at zero edge width.
        "lines.markeredgewidth": 0.9,
        # Type 42 (TrueType) rather than Type 3: required by most publishers and
        # keeps the text selectable and searchable in the final PDF.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save(fig, path, also_png: bool = True) -> None:
    """Write a figure as PDF, and optionally PNG for quick inspection."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"  Saved: {path}")
    if also_png:
        png = path.with_suffix(".png")
        fig.savefig(png, dpi=300)
        print(f"  Saved: {png}")
