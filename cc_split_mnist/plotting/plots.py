"""All figures, saved as PNG into <results>/<run_name>/plots/."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, name), dpi=150)
    plt.close(fig)


def _heat(ax, M, title, xlabel, ylabel, cmap="viridis", fmt="{:.2f}", annotate=True):
    im = ax.imshow(M, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if annotate and M.shape[0] <= 12 and M.shape[1] <= 12:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                        color="w", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)


def plot_accuracy_matrix(R: np.ndarray, outdir, name="acc_matrix.png", title=""):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    _heat(ax, R, f"Accuracy matrix R[i,j]\n(row=after task i, col=eval task j) {title}",
          "eval task", "trained through task")
    _save(fig, outdir, name)


def plot_acc_over_time(R: np.ndarray, outdir, name="acc_over_time.png", title=""):
    T = R.shape[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    for j in range(T):
        ax.plot(range(T), R[:, j], marker="o", label=f"task {j}")
    ax.plot(range(T), R.mean(axis=1), "k--", lw=2, label="mean")
    ax.set_xlabel("trained through task")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Per-task accuracy during the sequence {title}")
    ax.legend(fontsize=8)
    _save(fig, outdir, name)


def plot_forgetting(summary: Dict, outdir, name="forgetting.png"):
    f = summary["forgetting_per_task"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(range(len(f)), f, color="tomato")
    ax.set_xlabel("task")
    ax.set_ylabel("forgetting")
    ax.set_title(f"Forgetting per task (avg={summary['avg_forgetting']:.3f})")
    _save(fig, outdir, name)


def plot_method_comparison(results: Dict[str, np.ndarray], rand_acc, outdir,
                           name="method_comparison.png"):
    from metrics.continual_metrics import cl_summary
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    keys = ["ACC", "BWT", "avg_forgetting"]
    names = list(results.keys())
    for a, key in zip(axes, keys):
        vals = [cl_summary(results[m], rand_acc)[key] for m in names]
        a.bar(names, vals, color=["#777", "#4a7", "#47a"][: len(names)])
        a.set_title(key)
        a.axhline(0, color="k", lw=0.5)
    fig.suptitle("Backprop vs PC vs CC")
    _save(fig, outdir, name)


def plot_curves(results: Dict[str, np.ndarray], outdir, name="mean_acc_curves.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, R in results.items():
        ax.plot(range(R.shape[0]), R.mean(axis=1), marker="o", label=m)
    ax.set_xlabel("trained through task")
    ax.set_ylabel("mean accuracy over all tasks")
    ax.set_title("Average accuracy during the task sequence")
    ax.legend()
    _save(fig, outdir, name)


def plot_influence_by_task(infl: Dict[int, "np.ndarray"], outdir,
                           name="influence_by_task.png"):
    if not infl:
        return
    ts = sorted(infl)
    M = np.stack([np.asarray(infl[t]) for t in ts])
    M = M / (M.max(axis=1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    _heat(ax, M, "Normalized module do-influence on label, per task\n"
                 "(Schur-Fisher composed C^{l->L})", "module", "task", annotate=False)
    _save(fig, outdir, name)


def plot_supports(supports: Dict[int, "np.ndarray"], outdir, name="supports.png"):
    if not supports:
        return
    ts = sorted(supports)
    M = np.stack([np.asarray(supports[t], dtype=float) for t in ts])
    fig, ax = plt.subplots(figsize=(8, 3.0))
    _heat(ax, M, "Estimated task supports S_c (modules above influence quantile)",
          "module", "task", cmap="Greys", annotate=False)
    _save(fig, outdir, name)


def plot_confusion_graph(J: Optional[np.ndarray], outdir, name="support_overlap.png"):
    if J is None:
        return
    fig, ax = plt.subplots(figsize=(4.6, 4))
    _heat(ax, J, "Support overlap (Jaccard) ~ confusion graph", "task", "task",
          cmap="magma")
    _save(fig, outdir, name)


def plot_commutator(comm: Dict, outdir):
    for key, cmap in [("param", "plasma"), ("perf", "plasma")]:
        M = np.array(comm[key])
        fig, ax = plt.subplots(figsize=(4.6, 4))
        _heat(ax, M, f"Order-swap commutator proxy ({key} distance)\n"
                     "d(U_i U_j theta, U_j U_i theta)", "task", "task",
              cmap=cmap, fmt="{:.3f}")
        _save(fig, outdir, f"commutator_{key}.png")


def plot_leakage(leaks: Dict[int, Dict], outdir, name="leakage.png"):
    if not leaks:
        return
    ts = sorted(leaks)
    fr = [leaks[t]["eps_c_frac"] for t in ts]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(ts, fr, color="#47a")
    ax.set_xlabel("task")
    ax.set_ylabel("out-of-support update fraction")
    ax.set_title("Locality-error proxy eps_c (leakage outside support)")
    _save(fig, outdir, name)


def plot_gates(gate_snaps: Dict[int, List], outdir):
    for t, gates in gate_snaps.items():
        fig, axes = plt.subplots(1, len(gates), figsize=(4 * len(gates), 3.2))
        if len(gates) == 1:
            axes = [axes]
        for k, (ax, G) in enumerate(zip(axes, gates)):
            g = np.asarray(G)
            im = ax.imshow(g, cmap="coolwarm", aspect="auto", vmin=0, vmax=2)
            ax.set_title(f"layer {k} gate G (task {t})")
            plt.colorbar(im, ax=ax, fraction=0.046)
        _save(fig, outdir, f"gates_task{t}.png")


def plot_energy(traces: Dict[int, List[float]], outdir, name="free_energy.png"):
    if not traces:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for t, tr in sorted(traces.items()):
        ax.plot(tr, label=f"task {t}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean residual energy")
    ax.set_title("PC free-energy (residual) per epoch")
    ax.legend(fontsize=8)
    _save(fig, outdir, name)
