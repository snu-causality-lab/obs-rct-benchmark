"""Render the README overview heatmap from the published catalog-study scores.

The figure shows, for every catalog preset and recipe, the recipe's mean CATE MSE
relative to the RCT-only ExtraTrees T-learner in the same preset and budget. It reads
only `results/catalog_study/condition_scores.csv` and `not_applicable.csv`; it refits
nothing.

Usage: python scripts/render_overview.py [--output docs/figures]
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "results" / "catalog_study" / "condition_scores.csv"
TSR_SCORES = ROOT / "results" / "tsr_study" / "condition_scores.csv"  # same samples; identity checked in that study
REFERENCE = "rct_t_extra_trees"
BUDGETS = (("o512_r128", "OBS 512 · RCT 128"), ("o2048_r512", "OBS 2,048 · RCT 512"))
GROUPS = (
    ("RCT only", (("rct_t_ridge", "T ridge"), ("rct_dr_ridge", "DR ridge"), ("rct_dr_extra_trees", "DR trees"))),
    ("OBS only", (("obs_t_ridge", "T ridge"), ("obs_t_extra_trees", "T trees"))),
    ("Naive pooled", (("naive_pooled_t_ridge", "T ridge"), ("naive_pooled_t_extra_trees", "T trees"))),
    ("Fusion", (("cornet_official", "CorNet"), ("datafusionppi", "DataFusionPPI"))),
    ("TSR study", (("lmcgp_rct_only", "RCT GP"), ("tsr_tabpfn_lmcgp", "TSR"))),
)
PAPER_LABELS = {
    "cornet_official_n_conf": "CorNet design",
    "cvci_official_linear_bias": "CVCI linear bias",
    "elastic_integrative_gen_data_overlap": "Elastic integrative",
    "experimental_grounding_kps": "Experimental grounding",
    "integrative_r_learner_kps_code": "Integrative R (KPS code)",
    "integrative_r_learner_kps_normal_target_extension": "Integrative R (normal target)",
    "integrative_r_learner_multivariate_paper": "Integrative R (multivariate)",
    "power_likelihood_paper_text": "Power likelihood (paper)",
    "power_likelihood_supplement_code": "Power likelihood (supplement)",
}
CLIP = 3.0  # log2 ratio clipped to [1/8, 8]

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e", grid="#e4e3df", mid="#f0efec",
                  cool=("#104281", "#256abf", "#5598e7", "#9ec5f4"),
                  warm=("#f4b9b6", "#ec7f7c", "#d8383a", "#9c2224")),
    "dark": dict(surface="#1a1a19", text="#f0efec", muted="#c3c2b7", grid="#2e2e2c", mid="#383835",
                 cool=("#9ec5f4", "#6da7ec", "#3987e5", "#1f5aa3"),
                 warm=("#8f2a2b", "#c94546", "#e66767", "#f2a3a1")),
}


ROW_GROUPS = ((0, 12, "Structural\nmechanisms\nM0–M11"), (12, 22, "Paired\nvariants\nC01–C10"),
              (22, 31, "Paper-based\ndesigns"))


def row_key(scenario: str):
    track, name = scenario.split(":", 1)
    order = {"structural": 0, "addendum": 1, "paper": 2}[track]
    m = re.match(r"[A-Z](\d+)$", name)
    return (order, int(m.group(1)) if m else 0, name)


def row_label(scenario: str) -> str:
    track, name = scenario.split(":", 1)
    return PAPER_LABELS.get(name, name) if track == "paper" else name


def load():
    rows = list(csv.DictReader(SCORES.open()))
    table = {}
    for r in rows:
        preset, budget = r["scenario"].split("@")
        ok = r["applicable"] == "True" and int(r["successful"]) > 0
        table[(preset, budget, r["method"])] = dict(
            mean=float(r["mean"]) if ok else None,
            partial=ok and int(r["successful"]) < int(r["expected"]),
        )
    if TSR_SCORES.exists():
        for r in csv.DictReader(TSR_SCORES.open()):
            preset, budget = r["scenario"].split("@")
            ok = r["mean_mse"] not in ("", None)
            table[(preset, budget, r["method"])] = dict(
                mean=float(r["mean_mse"]) if ok else None,
                partial=ok and int(r["successful"]) < int(r["expected"]))
    presets = sorted({k[0] for k in table}, key=row_key)
    return table, presets


def cmap_for(theme):
    """Diverging blue (lower error) - neutral gray - red (higher error).

    Both themes run from the strongest pole to the neutral midpoint and back, so
    color intensity always grows with distance from "same"."""
    t = THEMES[theme]  # each theme lists its poles strongest-first (cool) and strongest-last (warm)
    return LinearSegmentedColormap.from_list(f"div-{theme}", list(t["cool"]) + [t["mid"]] + list(t["warm"]))


def render(theme: str, out: Path):
    t = THEMES[theme]
    table, presets = load()
    cols = [(g, m, lab) for g, items in GROUPS for m, lab in items]
    n_rows, n_cols = len(presets), len(cols)
    cmap = cmap_for(theme)
    plt.rcParams.update({"svg.fonttype": "none", "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 8.5, "text.color": t["text"]})
    cell_w, cell_h = 0.34, 0.2
    left, gap, right, top, bottom = 2.0, 0.35, 1.2, 1.45, 0.8
    panel_w = n_cols * cell_w
    fig_w = left + 2 * panel_w + gap + right
    fig_h = top + n_rows * cell_h + bottom
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=t["surface"])
    for p, (budget, title) in enumerate(BUDGETS):
        x0 = (left + p * (panel_w + gap)) / fig_w
        ax = fig.add_axes([x0, bottom / fig_h, panel_w / fig_w, n_rows * cell_h / fig_h])
        ax.set_facecolor(t["surface"])
        ax.set_xlim(0, n_cols)
        ax.set_ylim(n_rows, 0)
        for i, preset in enumerate(presets):
            ref = table.get((preset, budget, REFERENCE), {}).get("mean")
            for j, (_, method, _) in enumerate(cols):
                cell = table.get((preset, budget, method))
                if cell is None or cell["mean"] is None or not ref:
                    ax.text(j + 0.5, i + 0.5, "–", ha="center", va="center", color=t["muted"], fontsize=7)
                    continue
                v = max(-CLIP, min(CLIP, math.log2(cell["mean"] / ref)))
                ax.add_patch(plt.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, color=cmap((v + CLIP) / (2 * CLIP)), lw=0))
                if cell["partial"]:
                    ax.text(j + 0.5, i + 0.5, "•", ha="center", va="center", color=t["text"], fontsize=7)
        for boundary in (12, 22):
            ax.axhline(boundary, color=t["muted"], lw=0.6)
        if p == 1:  # name the row groups once, to the right of the second panel
            for lo, hi, name in ROW_GROUPS:
                ax.text(n_cols + 0.4, (lo + hi) / 2, name, ha="left", va="center", fontsize=7.5,
                        color=t["text"], fontweight="bold", linespacing=1.3)
        start = 0
        for g, items in GROUPS:
            if start:
                ax.axvline(start, color=t["surface"], lw=2.5)
            ax.text(start + len(items) / 2, -3.9, g, ha="center", va="bottom", fontsize=8, color=t["text"], fontweight="bold")
            ax.plot([start + 0.1, start + len(items) - 0.1], [-3.75, -3.75], color=t["muted"], lw=0.6, clip_on=False)
            start += len(items)
        ax.set_xticks([j + 0.5 for j in range(n_cols)])
        ax.set_xticklabels([lab for _, _, lab in cols], rotation=50, ha="left", rotation_mode="anchor", fontsize=7.5, color=t["muted"])
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=2)
        ax.set_yticks([i + 0.5 for i in range(n_rows)])
        ax.set_yticklabels([row_label(s) for s in presets] if p == 0 else [], fontsize=7.5, color=t["text"])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(n_cols / 2, -5.6, title, ha="center", va="bottom", fontsize=9.5, color=t["text"], fontweight="bold")
    # colorbar legend
    cax = fig.add_axes([(left + 0.2) / fig_w, 0.22 / fig_h, (2 * panel_w + gap - 0.4) / fig_w, 0.12 / fig_h])
    import numpy as np
    grad = np.linspace(0, 1, 256)[None, :]
    cax.imshow(grad, aspect="auto", cmap=cmap, extent=(-CLIP, CLIP, 0, 1))
    cax.set_yticks([])
    ticks = [-3, -2, -1, 0, 1, 2, 3]
    cax.set_xticks(ticks)
    cax.set_xticklabels(["8× lower", "4×", "2×", "same", "2×", "4×", "8× higher"], fontsize=7.5, color=t["muted"])
    cax.tick_params(length=0, pad=2)
    for s in cax.spines.values():
        s.set_visible(False)
    fig.text((left + 0.2) / fig_w, 0.52 / fig_h,
             "Mean CATE MSE relative to the RCT-only ExtraTrees T-learner (the reference, not shown; blue = lower error)   •  some seeds failed   –  not applicable",
             fontsize=7.5, color=t["muted"], va="bottom")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=t["surface"], metadata={"Date": None})
    plt.close(fig)


RCT_ONLY = ("rct_t_ridge", "rct_t_extra_trees", "rct_dr_ridge", "rct_dr_extra_trees")
TABLE_ROWS = (("tsr_tabpfn_lmcgp", "TSR (reimplemented)"), ("lmcgp_rct_only", "RCT-only GP (TSR control)"),
              ("cornet_official", "CorNet"), ("datafusionppi", "DataFusionPPI"),
              ("naive_pooled_t_ridge", "Naive pooled T ridge"), ("naive_pooled_t_extra_trees", "Naive pooled T trees"),
              ("obs_t_ridge", "OBS-only T ridge"), ("obs_t_extra_trees", "OBS-only T trees"),
              ("rct_t_ridge", "RCT-only T ridge"), ("rct_dr_ridge", "RCT-only DR ridge"),
              ("rct_dr_extra_trees", "RCT-only DR trees"))


def count_table() -> str:
    """Markdown counts: presets where a recipe's mean MSE is below (a) the RCT-only ExtraTrees
    T-learner, one of the catalog study's two prespecified controls, and (b) every
    RCT-only recipe. Descriptive; no uncertainty."""
    table, presets = load()
    head = ["Recipe"] + [f"vs reference, {t}" for _, t in BUDGETS] + [f"vs all RCT-only, {t}" for _, t in BUDGETS]
    lines = ["| " + " | ".join(head) + " |", "| --- |" + " ---: |" * (len(head) - 1)]
    for method, label in TABLE_ROWS:
        cells = []
        for strict in (False, True):
            for budget, _ in BUDGETS:
                if strict and method in RCT_ONLY:
                    cells.append("—")
                    continue
                won = n = partial = 0
                for p in presets:
                    c = table.get((p, budget, method))
                    refs = [table.get((p, budget, m), {}).get("mean") for m in (RCT_ONLY if strict else (REFERENCE,))]
                    if c is None or c["mean"] is None or any(r is None for r in refs):
                        continue
                    n += 1
                    won += c["mean"] < min(refs)
                    partial += c["partial"]
                cells.append(f"{won}/{n}" + ("†" if partial else ""))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / "docs" / "figures"))
    a = ap.parse_args()
    for theme in THEMES:
        render(theme, Path(a.output) / f"catalog_relative_mse_{theme}.svg")
    (Path(a.output) / "catalog_relative_mse_counts.md").write_text(count_table())


if __name__ == "__main__":
    main()
