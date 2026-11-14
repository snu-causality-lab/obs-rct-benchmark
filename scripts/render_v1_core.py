"""Render the README overview of the v1 core study from its published packet.

Each cell is a recipe's mean CATE MSE relative to `rct_selected`, the RCT-only
reference chosen from trial data, in the same preset and budget. A dot marks a
Holm-significant difference in the declared analysis (docs/V1_CORE_STUDY.md).
It reads only results/v1_core/condition_scores.csv, primary_contrasts.csv and
analysis.json; it refits nothing.

Usage: python scripts/render_v1_core.py [--output docs/figures]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from render_overview import CLIP, ROW_GROUPS, THEMES, cmap_for, row_key, row_label

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "results" / "v1_core"
REFERENCE = "rct_selected"
RCT_ONLY = {"rct_t_ridge", "rct_t_extra_trees", "rct_dr_ridge", "rct_dr_extra_trees"}
BUDGETS = (("o512_r128", "OBS 512 · RCT 128"), ("o2048_r512", "OBS 2,048 · RCT 512"))
GROUPS = (
    ("RCT only", (("rct_t_ridge", "T ridge"), ("rct_t_extra_trees", "T trees"),
                  ("rct_dr_ridge", "DR ridge"), ("rct_dr_extra_trees", "DR trees"))),
    ("OBS only", (("obs_t_ridge", "T ridge"), ("obs_t_extra_trees", "T trees"))),
    ("Naive pooled", (("naive_pooled_t_ridge", "T ridge"), ("naive_pooled_t_extra_trees", "T trees"))),
    ("Fusion", (("cornet_raw", "CorNet raw"), ("cornet_official", "CorNet std."),
                ("datafusionppi", "DataFusionPPI"))),
)
LABELS = {"cornet_raw": "CorNet, raw inputs", "cornet_official": "CorNet, standardized inputs",
          "datafusionppi": "DataFusionPPI", "obs_t_ridge": "OBS-only T ridge",
          "obs_t_extra_trees": "OBS-only T trees", "naive_pooled_t_ridge": "Naive pooled T ridge",
          "naive_pooled_t_extra_trees": "Naive pooled T trees", "rct_t_ridge": "RCT-only T ridge",
          "rct_t_extra_trees": "RCT-only T trees (catalog reference)",
          "rct_dr_ridge": "RCT-only DR ridge", "rct_dr_extra_trees": "RCT-only DR trees"}


def load():
    means = {}
    for r in csv.DictReader((PACKET / "condition_scores.csv").open()):
        preset, budget = r["scenario"].split("@")
        means[(preset, budget, r["method"])] = (float(r["mean_mse"]) if r["mean_mse"] else None,
                                                int(r["successful"]) < int(r["expected"]))
    verdicts = {}
    for r in csv.DictReader((PACKET / "primary_contrasts.csv").open()):
        preset, budget = r["scenario"].split("@")
        verdicts[(preset, budget, r["family"])] = r["verdict"]
    presets = sorted({k[0] for k in means}, key=row_key)
    return means, verdicts, presets


def verdict_for(verdicts, preset, budget, method):
    if method in RCT_ONLY:  # RCT-only families are declared on the OBS 512 records
        budget = "o512_" + budget.split("_")[1]
    return verdicts.get((preset, budget, method))


def render(theme: str, out: Path) -> None:
    t = THEMES[theme]
    means, verdicts, presets = load()
    cols = [(m, lab) for _, items in GROUPS for m, lab in items]
    n_rows, n_cols = len(presets), len(cols)
    cmap = cmap_for(theme)
    plt.rcParams.update({"svg.fonttype": "none", "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 8.5, "text.color": t["text"]})
    cell_w, cell_h = 0.34, 0.2
    left, gap, right, top, bottom = 2.0, 0.35, 1.2, 1.45, 0.8
    panel_w = n_cols * cell_w
    fig_w, fig_h = left + 2 * panel_w + gap + right, top + n_rows * cell_h + bottom
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=t["surface"])
    for p, (budget, title) in enumerate(BUDGETS):
        ax = fig.add_axes([(left + p * (panel_w + gap)) / fig_w, bottom / fig_h,
                           panel_w / fig_w, n_rows * cell_h / fig_h])
        ax.set_facecolor(t["surface"])
        ax.set_xlim(0, n_cols)
        ax.set_ylim(n_rows, 0)
        for i, preset in enumerate(presets):
            ref = (means.get((preset, budget, REFERENCE)) or (None, False))[0]
            for j, (method, _) in enumerate(cols):
                mean, partial = means.get((preset, budget, method), (None, False))
                if mean is None or not ref:
                    ax.text(j + 0.5, i + 0.5, "–", ha="center", va="center", color=t["muted"], fontsize=7)
                    continue
                v = max(-CLIP, min(CLIP, math.log2(mean / ref)))
                ax.add_patch(plt.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88,
                                           color=cmap((v + CLIP) / (2 * CLIP)), lw=0))
                verdict = verdict_for(verdicts, preset, budget, method)
                mark = {"better": "•", "worse": "•", "untested": "×"}.get(verdict)
                if mark:
                    ax.text(j + 0.5, i + 0.5, mark, ha="center", va="center", color=t["text"], fontsize=7)
        for boundary in (12, 22):
            ax.axhline(boundary, color=t["muted"], lw=0.6)
        if p == 1:
            for lo, hi, name in ROW_GROUPS:
                ax.text(n_cols + 0.4, (lo + hi) / 2, name, ha="left", va="center", fontsize=7.5,
                        color=t["text"], fontweight="bold", linespacing=1.3)
        start = 0
        for g, items in GROUPS:
            if start:
                ax.axvline(start, color=t["surface"], lw=2.5)
            ax.text(start + len(items) / 2, -3.9, g, ha="center", va="bottom", fontsize=8,
                    color=t["text"], fontweight="bold")
            ax.plot([start + 0.1, start + len(items) - 0.1], [-3.75, -3.75], color=t["muted"],
                    lw=0.6, clip_on=False)
            start += len(items)
        ax.set_xticks([j + 0.5 for j in range(n_cols)])
        ax.set_xticklabels([lab for _, lab in cols], rotation=50, ha="left", rotation_mode="anchor",
                           fontsize=7.5, color=t["muted"])
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=2)
        ax.set_yticks([i + 0.5 for i in range(n_rows)])
        ax.set_yticklabels([row_label(s) for s in presets] if p == 0 else [], fontsize=7.5, color=t["text"])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(n_cols / 2, -5.6, title, ha="center", va="bottom", fontsize=9.5, color=t["text"],
                fontweight="bold")
    cax = fig.add_axes([(left + 0.2) / fig_w, 0.22 / fig_h, (2 * panel_w + gap - 0.4) / fig_w, 0.12 / fig_h])
    cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cmap, extent=(-CLIP, CLIP, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    cax.set_xticklabels(["8× lower", "4×", "2×", "same", "2×", "4×", "8× higher"], fontsize=7.5, color=t["muted"])
    cax.tick_params(length=0, pad=2)
    for s in cax.spines.values():
        s.set_visible(False)
    fig.text((left + 0.2) / fig_w, 0.52 / fig_h,
             "Mean CATE MSE relative to rct_selected (RCT-only, chosen from trial data; blue = lower error)"
             "   •  Holm-significant   ×  not tested (failed seeds)   –  not applicable",
             fontsize=7.5, color=t["muted"], va="bottom")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=t["surface"], metadata={"Date": None})
    plt.close(fig)


def count_table() -> str:
    """Markdown headline from analysis.json: verdict counts per recipe family."""
    families = json.loads((PACKET / "analysis.json").read_text())["families"]
    order = ["cornet_raw", "cornet_official", "datafusionppi", "naive_pooled_t_ridge",
             "naive_pooled_t_extra_trees", "obs_t_ridge", "obs_t_extra_trees",
             "rct_t_ridge", "rct_t_extra_trees", "rct_dr_ridge", "rct_dr_extra_trees"]
    lines = ["| Recipe | Conditions | Better | Worse | Unresolved | Untested | Transport conditions: better / worse |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m in order:
        f = families[m]
        total = {v: f[f"other_{v}"] + f[f"transport_{v}"] for v in ("better", "worse", "unresolved", "untested")}
        transport = sum(f[f"transport_{v}"] for v in ("better", "worse", "unresolved", "untested"))
        lines.append(f"| {LABELS[m]} | {f['size']} | {total['better']} | {total['worse']} | {total['unresolved']} "
                     f"| {total['untested']} | {f['transport_better']} / {f['transport_worse']} of {transport} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / "docs" / "figures"))
    a = ap.parse_args()
    for theme in THEMES:
        render(theme, Path(a.output) / f"v1_core_relative_mse_{theme}.svg")
    (Path(a.output) / "v1_core_counts.md").write_text(count_table())


if __name__ == "__main__":
    main()
