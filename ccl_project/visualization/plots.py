"""
visualization/plots.py
All plotting for the CCL Split-MNIST experiment.

Figures produced:
  comparison.png        — 2x3 grid: heatmaps + curves + BWT bars
  per_task_accuracy.png — one subplot per task showing each method's forgetting
  causal_gates.png      — gate heatmaps for PC+CC (layer 1 and layer 2)
"""

from __future__ import annotations
import os
from typing import List

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from configs import CFG

# ── Palette ──────────────────────────────────────────────────────────────────
BG       = "#0d0d17"
PANEL_BG = "#12122a"
GRID_C   = "#1c1c38"
AXIS_C   = "#28285a"

C_NAIVE = "#ff4d6d"
C_PC    = "#4db8ff"
C_CC    = "#39ff9e"
COLORS  = [C_NAIVE, C_PC, C_CC]

_heat_cmap = LinearSegmentedColormap.from_list(
    "ccl_heat", ["#2b0010", "#8b0000", "#e67e00", "#27ae60", "#0a3d1f"], N=256
)
_gate_cmap = LinearSegmentedColormap.from_list(
    "ccl_gate", ["#0d0d17", "#1a1040", "#6a0dad", "#ff9900", "#fff8dc"], N=256
)


def _ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="#9999bb", labelsize=8)
    ax.xaxis.label.set_color("#9999bb")
    ax.yaxis.label.set_color("#9999bb")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor(AXIS_C)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=7)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, color=GRID_C, linestyle="--", linewidth=0.5, alpha=0.8)


def _heatmap(ax, acc, title, task_names, title_color="white"):
    n = len(task_names)
    mat = np.where(np.tril(np.ones((n, n), dtype=bool)), acc, np.nan)
    im = ax.imshow(mat, vmin=50, vmax=100, cmap=_heat_cmap, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"T{j+1}" for j in range(n)], color="#9999bb", fontsize=8)
    ax.set_yticklabels([f"After T{i+1}" for i in range(n)], color="#9999bb", fontsize=8)
    ax.set_title(title, color=title_color, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Evaluated on →", color="#9999bb", fontsize=7)
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(AXIS_C)
    for i in range(n):
        for j in range(n):
            if j <= i:
                v = acc[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="black" if v > 75 else "white",
                        fontsize=9, fontweight="bold")
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(colors="#9999bb", labelsize=7)
    cb.outline.set_edgecolor(AXIS_C)


# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison(results, task_names, output_dir=CFG.OUTPUT_DIR,
                    save=CFG.SAVE_FIG, show=CFG.SHOW_FIG):
    """
    Main 2×3 comparison figure.
    Row 1: accuracy heatmaps  (Naive | PC | PC+CC)
    Row 2: avg accuracy curve | task-1 forgetting curve | BWT bar chart
    """
    assert len(results) == 3
    n = len(task_names)

    fig = plt.figure(figsize=(20, 12), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.46, wspace=0.30,
                             top=0.91, bottom=0.07, left=0.06, right=0.97)

    # ── Row 1: heatmaps ───────────────────────────────────────────
    for col, (res, col_c) in enumerate(zip(results, COLORS)):
        ax = fig.add_subplot(gs[0, col])
        _heatmap(ax, res.acc_matrix, res.name, task_names, title_color=col_c)

    # ── Row 2-left: average accuracy over time ────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    _ax_style(ax4, "Average Accuracy Over Time",
              "Tasks Trained So Far", "Avg Accuracy on Seen Tasks (%)")
    for res, c in zip(results, COLORS):
        ys = [res.avg_accuracy_after(t) for t in range(n)]
        ax4.plot(range(1, n+1), ys, "o-", color=c, label=res.name,
                 lw=2.2, ms=7, markeredgecolor=BG, markeredgewidth=1.2)
    ax4.set_xlim(0.7, n + 0.3)
    ax4.set_ylim(45, 103)
    ax4.set_xticks(range(1, n+1))
    ax4.legend(facecolor="#191930", edgecolor=AXIS_C,
               labelcolor="white", fontsize=8, framealpha=0.9)

    # ── Row 2-mid: task-1 forgetting curve ───────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    _ax_style(ax5, "Task 1 Forgetting Curve",
              "Tasks Trained So Far", "Task 1 Accuracy (%)")
    for res, c in zip(results, COLORS):
        ax5.plot(range(1, n+1), res.acc_matrix[:, 0], "o-", color=c,
                 label=res.name, lw=2.2, ms=7,
                 markeredgecolor=BG, markeredgewidth=1.2)
    ax5.set_xlim(0.7, n + 0.3)
    ax5.set_ylim(45, 103)
    ax5.set_xticks(range(1, n+1))

    # Annotate final drop for each method
    for res, c in zip(results, COLORS):
        drop  = res.forgetting(0)
        final = res.acc_matrix[-1, 0]
        sign  = "↓" if drop < 0 else "↑"
        ax5.annotate(f"{sign}{abs(drop):.1f}%",
                     xy=(n, final),
                     xytext=(n - 0.9, final + (8 if final < 75 else -10)),
                     color=c, fontsize=8, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=c, lw=0.9))
    ax5.legend(facecolor="#191930", edgecolor=AXIS_C,
               labelcolor="white", fontsize=8, framealpha=0.9)

    # ── Row 2-right: BWT bar chart ────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    _ax_style(ax6, "Backward Transfer\n(↑ closer to 0 = less forgetting)",
              ylabel="BWT (%)")
    ax6.grid(True, axis="y", color=GRID_C, linestyle="--", linewidth=0.5)
    ax6.grid(False, axis="x")

    bwts   = [r.backward_transfer() for r in results]
    labels = ["Naive\nMLP", "PC\nNetwork", "PC +\nCC"]
    xs     = np.arange(3)
    bars   = ax6.bar(xs, bwts, width=0.5, color=COLORS,
                     edgecolor=BG, linewidth=1.1)
    ax6.axhline(0, color="#9999bb", lw=0.8, linestyle="--")
    ax6.set_xticks(xs)
    ax6.set_xticklabels(labels, color="white", fontsize=9)
    for bar, v in zip(bars, bwts):
        ypos = v - 2.0 if v < 0 else v + 0.3
        ax6.text(bar.get_x() + bar.get_width() / 2, ypos,
                 f"{v:.1f}%", ha="center", color="white",
                 fontsize=10, fontweight="bold")

    # ── Stat box (top-right corner) ───────────────────────────────
    stats_text = "Final avg accuracy\n" + "\n".join(
        f"  {r.name}: {r.final_avg_accuracy():.1f}%" for r in results
    )
    fig.text(0.98, 0.97, stats_text, ha="right", va="top",
             color="#ccccee", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#191930",
                       edgecolor=AXIS_C, alpha=0.9))

    fig.suptitle(
        "Split MNIST — Continual Learning Comparison\n"
        "Naive MLP  ·  Discriminative PC Network  ·  PC + Causal Coding",
        color="white", fontsize=13, fontweight="bold", y=0.97,
    )

    if save:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "comparison.png")
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
        print(f"  ✓  comparison.png  →  {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_gates(gate_history_l1, gate_history_l2, task_names,
               output_dir=CFG.OUTPUT_DIR, save=CFG.SAVE_FIG, show=CFG.SHOW_FIG):
    """
    Gate heatmaps for PC+CC.
    Bright = high causal influence = module is in task's support S_t.
    Disjoint bright patterns → nearly disjoint supports → low commutators.
    """
    n_tasks   = len(task_names)
    n_modules = len(gate_history_l1[0])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor=BG)

    for ax, gates, layer_lbl in zip(axes,
                                     [gate_history_l1, gate_history_l2],
                                     ["Layer 1  —  Causal Gates", "Layer 2  —  Causal Gates"]):
        mat = np.array(gates)   # (n_tasks, n_modules)
        im  = ax.imshow(mat, cmap=_gate_cmap, vmin=0, aspect="auto")
        ax.set_xticks(range(n_modules))
        ax.set_xticklabels([f"Module {m+1}" for m in range(n_modules)],
                            color="#9999bb", fontsize=9)
        ax.set_yticks(range(n_tasks))
        ax.set_yticklabels([f"T{t+1}  ({task_names[t]})" for t in range(n_tasks)],
                            color="#9999bb", fontsize=9)
        ax.set_title(layer_lbl, color="white", fontsize=11, fontweight="bold")
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(AXIS_C)
        for i in range(n_tasks):
            for j in range(n_modules):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="#00ffcc", fontsize=9, fontweight="bold")
        cb = plt.colorbar(im, ax=ax, fraction=0.055, pad=0.02)
        cb.ax.tick_params(colors="#9999bb", labelsize=7)
        cb.outline.set_edgecolor(AXIS_C)

    fig.suptitle(
        "Causal Gates  ·  PC + Causal Coding\n"
        "Bright = high do-influence  ·  Disjoint patterns → small commutators → less forgetting",
        color="white", fontsize=10, fontweight="bold", y=1.04,
    )
    fig.patch.set_facecolor(BG)
    plt.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "causal_gates.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  ✓  causal_gates.png  →  {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_per_task_accuracy(results, task_names,
                           output_dir=CFG.OUTPUT_DIR,
                           save=CFG.SAVE_FIG, show=CFG.SHOW_FIG):
    """
    One subplot per task showing each method's accuracy on that task
    as more tasks are trained sequentially.
    Reveals exactly when and how much each method forgets each task.
    """
    n    = len(task_names)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 4.8, rows * 3.8),
                              facecolor=BG)
    axes = np.array(axes).flatten()

    for t in range(n):
        ax = axes[t]
        _ax_style(ax,
                  title=f"Task {t+1}  ({task_names[t]})",
                  xlabel="Tasks trained so far",
                  ylabel="Accuracy (%)")

        for res, c in zip(results, COLORS):
            xs = list(range(t + 1, n + 1))
            ys = [res.acc_matrix[i, t] for i in range(t, n)]
            ax.plot(xs, ys, "o-", color=c, label=res.name,
                    lw=2.0, ms=6, markeredgecolor=BG, markeredgewidth=1.0)

            # Shade forgetting region
            if len(ys) > 1:
                ax.fill_between(xs, ys[0], ys, alpha=0.06, color=c)

        ax.axhline(ys[0] if len(ys) > 0 else 90,
                   color="#333355", linestyle=":", linewidth=0.9)
        ax.set_xlim(t + 0.7, n + 0.3)
        ax.set_ylim(42, 103)
        ax.set_xticks(range(t + 1, n + 1))
        if t == 0:
            ax.legend(facecolor="#191930", edgecolor=AXIS_C,
                      labelcolor="white", fontsize=7, framealpha=0.9)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "Per-Task Accuracy as More Tasks Are Trained\n"
        "(each panel = one task;  x-axis = tasks trained so far;  "
        "y-axis = accuracy on that task)",
        color="white", fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "per_task_accuracy.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  ✓  per_task_accuracy.png  →  {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
