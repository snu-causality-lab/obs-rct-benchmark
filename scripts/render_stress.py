"""Render the stress-study dose-response figure from its published packet.

Each panel is one dial at one trial size. Lines show a recipe's mean CATE MSE divided
by that of `rct_selected` at each level (log scale; below 1 is better than the
reference). A filled marker is a Holm-significant difference in the declared
analysis; a hollow marker is unresolved. Reads only results/stress_cornet/
dose_response.csv and contrasts.csv.

Usage: python scripts/render_stress.py [--output docs/figures]
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "results" / "stress_cornet"
DIALS = (("hidden_modifier", "Hidden effect\nmodifier"), ("bias_complexity", "Bias\nfrequency"),
         ("common_shift", "Per-arm\nshift"), ("obs_selection", "OBS treatment\nselection"),
         ("obs_size", "Fewer OBS\nrows"), ("noise_shape", "Unequal noise\nvariance"),
         ("trial_shrink", "Narrower trial*"), ("target_shift", "Shifted\ntarget*"))
# Fixed categorical order (validated with the dataviz palette checker in both modes).
SERIES = (("cornet_raw", "CorNet, raw inputs", "o"), ("cornet_official", "CorNet, standardized", "s"),
          ("datafusionppi", "DataFusionPPI", "D"), ("obs_t_ridge", "OBS-only T ridge", "^"),
          ("naive_pooled_t_ridge", "Naive pooled T ridge", "v"))
THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e", grid="#e4e3df",
                  series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")),
    "dark": dict(surface="#1a1a19", text="#f0efec", muted="#c3c2b7", grid="#2e2e2c",
                 series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181")),
}


def load():
    ratio = {}
    for r in csv.DictReader((PACKET / "dose_response.csv").open()):
        ratio[(r["method"], r["dial"], int(r["level"]), int(r["rct"]))] = float(r["ratio_of_means"])
    verdict = {}
    for r in csv.DictReader((PACKET / "contrasts.csv").open()):
        verdict[(r["family"], r["dial"], int(r["level"]), int(r["rct"]))] = r["verdict"]
    return ratio, verdict


def render(theme: str, out: Path) -> None:
    t = THEMES[theme]
    ratio, verdict = load()
    plt.rcParams.update({"svg.fonttype": "none", "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 8, "text.color": t["text"]})
    fig, axes = plt.subplots(2, len(DIALS), figsize=(13.2, 4.9), sharey=True, facecolor=t["surface"])
    for row, rct in enumerate((128, 512)):
        for col, (dial, label) in enumerate(DIALS):
            ax = axes[row][col]
            ax.set_facecolor(t["surface"])
            ax.axhline(0, color=t["muted"], lw=0.8)
            for (method, name, marker), color in zip(SERIES, t["series"]):
                ys = [math.log2(ratio[(method, dial, lvl, rct)]) for lvl in range(4)]
                ax.plot(range(4), ys, color=color, lw=1.6, zorder=2)
                for lvl, y in enumerate(ys):
                    decisive = verdict.get((method, dial, lvl, rct)) in ("better", "worse")
                    ax.plot(lvl, y, marker=marker, ms=4.5, color=color, zorder=3,
                            markerfacecolor=color if decisive else t["surface"], markeredgewidth=1.2)
            ax.set_ylim(-4.2, 3.6)
            ax.set_yticks([-4, -2, -1, 0, 1, 2, 3])
            ax.set_yticklabels(["1/16×", "1/4×", "1/2×", "same", "2×", "4×", "8×"], color=t["muted"])
            ax.set_xticks(range(4))
            ax.set_xticklabels(["0", "1", "2", "3"], color=t["muted"])
            ax.grid(axis="y", color=t["grid"], lw=0.6)
            ax.tick_params(length=0, pad=2)
            for s in ax.spines.values():
                s.set_visible(False)
            if row == 0:
                ax.set_title(label, fontsize=8.5, color=t["text"], fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"RCT {rct}\nerror vs reference", color=t["text"])
            if row == 1:
                ax.set_xlabel("level", color=t["muted"])
    handles = [plt.Line2D([], [], color=c, marker=m, lw=1.6, ms=5, label=n)
               for (_, n, m), c in zip(SERIES, t["series"])]
    fig.legend(handles=handles, loc="lower center", ncol=len(SERIES), frameon=False, fontsize=8,
               labelcolor=t["text"], bbox_to_anchor=(0.5, -0.005))
    fig.text(0.5, 0.075, "Mean CATE MSE relative to rct_selected (log scale; below 'same' is better).  "
             "Filled marker: Holm-significant; hollow: unresolved.  * extrapolation dial",
             ha="center", fontsize=7.5, color=t["muted"])
    fig.subplots_adjust(left=0.07, right=0.995, top=0.88, bottom=0.2, wspace=0.12, hspace=0.25)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=t["surface"], metadata={"Date": None})
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / "docs" / "figures"))
    a = ap.parse_args()
    for theme in THEMES:
        render(theme, Path(a.output) / f"stress_cornet_{theme}.svg")


if __name__ == "__main__":
    main()
