"""
plotting.py
===========
Every figure gets saved as a PNG under `<output_dir>/figures/`. Nothing is
shown interactively (this is meant to run headlessly). Each function
returns the path it wrote, for easy logging / inclusion in a report.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=0.95)
MODEL_COLORS = {"MLP": "#4C72B0", "PC": "#DD8452", "CC": "#55A868"}


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_curves(histories: dict, out_dir: str, key: str, ylabel: str, name: str, logy=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    for model_name, hist in histories.items():
        vals = hist.get(key)
        if vals is None or len(vals) == 0:
            continue
        ax.plot(range(1, len(vals) + 1), vals, marker="o", ms=3,
                label=model_name, color=MODEL_COLORS.get(model_name))
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.set_title(ylabel + " over training")
    ax.legend()
    return _save(fig, out_dir, name)


def plot_cc_phase_diagnostics(cc_history: dict, out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(cc_history["alpha_t"]) + 1)
    axes[0].plot(epochs, cc_history["alpha_t"], color="#55A868")
    axes[0].set_title(r"gate strength $\alpha_t$")
    axes[0].set_xlabel("epoch")
    axes[1].plot(epochs, cc_history["gate_mean"], label="mean", color="#55A868")
    axes[1].fill_between(epochs,
                          np.array(cc_history["gate_mean"]) - np.array(cc_history["gate_std"]),
                          np.array(cc_history["gate_mean"]) + np.array(cc_history["gate_std"]),
                          alpha=0.2, color="#55A868")
    axes[1].set_title(r"gate value $G_l$ (mean$\pm$std)")
    axes[1].set_xlabel("epoch")
    axes[2].plot(epochs, cc_history["gate_suppressed_frac"], color="#C44E52")
    axes[2].set_title("fraction of suppressed connections")
    axes[2].set_xlabel("epoch")
    fig.suptitle("CC schedule diagnostics (warmup -> soft-gate -> full-CC)")
    fig.tight_layout()
    return _save(fig, out_dir, "cc_phase_diagnostics")


def plot_confusion_matrices(cms: dict, out_dir: str):
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, cms.items()):
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        sns.heatmap(cm_norm, annot=False, cmap="Blues", ax=ax, cbar=ax is axes[-1])
        ax.set_title(f"{name} confusion matrix (row-normalized)")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    fig.tight_layout()
    return _save(fig, out_dir, "confusion_matrices")


def plot_weight_heatmaps(weight_dicts: dict, out_dir: str, layer_idx: int = 0):
    n = len(weight_dicts)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, (name, Ws) in zip(axes, weight_dicts.items()):
        W = Ws[layer_idx]
        vmax = np.percentile(np.abs(W), 99) + 1e-9
        sns.heatmap(W, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax, ax=ax, cbar=ax is axes[-1])
        ax.set_title(f"{name} layer {layer_idx} weights")
        ax.set_xlabel("input unit")
        ax.set_ylabel("output unit")
    fig.tight_layout()
    return _save(fig, out_dir, f"weight_heatmaps_layer{layer_idx}")


def plot_sparsity_bars(sparsity_reports: dict, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    model_names = list(sparsity_reports.keys())
    layers = list(range(len(next(iter(sparsity_reports.values()))["per_layer"])))
    width = 0.8 / len(model_names)
    for i, name in enumerate(model_names):
        per_layer = sparsity_reports[name]["per_layer"]
        xs = np.arange(len(layers)) + i * width
        axes[0].bar(xs, [p["sparsity"] for p in per_layer], width=width,
                    label=name, color=MODEL_COLORS.get(name))
        axes[1].bar(xs, [p["effective_connectivity"] for p in per_layer], width=width,
                    label=name, color=MODEL_COLORS.get(name))
    axes[0].set_title("weight sparsity (% near-zero) per layer")
    axes[0].set_xlabel("layer")
    axes[0].set_xticks(np.arange(len(layers)) + width)
    axes[0].set_xticklabels(layers)
    axes[1].set_title("effective connectivity (active inputs / output unit)")
    axes[1].set_xlabel("layer")
    axes[1].set_xticks(np.arange(len(layers)) + width)
    axes[1].set_xticklabels(layers)
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    return _save(fig, out_dir, "sparsity_connectivity")


def plot_effective_rank(sparsity_reports: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    model_names = list(sparsity_reports.keys())
    layers = list(range(len(next(iter(sparsity_reports.values()))["per_layer"])))
    width = 0.8 / len(model_names)
    for i, name in enumerate(model_names):
        per_layer = sparsity_reports[name]["per_layer"]
        xs = np.arange(len(layers)) + i * width
        ax.bar(xs, [p["effective_rank"] for p in per_layer], width=width,
               label=name, color=MODEL_COLORS.get(name))
    ax.set_title("effective rank per weight matrix (lower = more structured)")
    ax.set_xlabel("layer")
    ax.set_xticks(np.arange(len(layers)) + width)
    ax.set_xticklabels(layers)
    ax.legend()
    fig.tight_layout()
    return _save(fig, out_dir, "effective_rank")


def plot_entanglement_bars(ent_reports: dict, out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    names = list(ent_reports.keys())
    colors = [MODEL_COLORS.get(n) for n in names]
    axes[0].bar(names, [ent_reports[n]["mean_entanglement"] for n in names], color=colors)
    axes[0].set_title("mean # classes/neuron respond strongly to\n(lower = less entangled)")
    axes[1].bar(names, [ent_reports[n]["mean_neuron_label_mi"] for n in names], color=colors)
    axes[1].set_title("mean neuron-label mutual information\n(higher = more label-informative neurons)")
    axes[2].bar(names, [ent_reports[n]["mean_abs_pairwise_corr"] for n in names], color=colors)
    axes[2].set_title("mean |pairwise neuron correlation|\n(lower = less redundant)")
    fig.tight_layout()
    return _save(fig, out_dir, "entanglement_summary")


def plot_entanglement_histograms(ent_reports: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, rep in ent_reports.items():
        ax.hist(rep["per_neuron_n_classes"], bins=np.arange(0, 12) - 0.5, alpha=0.5,
                label=name, color=MODEL_COLORS.get(name), density=True)
    ax.set_xlabel("# classes a neuron responds strongly to")
    ax.set_ylabel("fraction of neurons")
    ax.set_title("per-neuron entanglement distribution")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out_dir, "entanglement_histogram")


def plot_correlation_heatmaps(ent_reports: dict, out_dir: str):
    n = len(ent_reports)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, (name, rep) in zip(axes, ent_reports.items()):
        sns.heatmap(rep["corr_matrix"], cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax,
                    cbar=ax is axes[-1], xticklabels=False, yticklabels=False)
        ax.set_title(f"{name} neuron-neuron activation correlation")
    fig.tight_layout()
    return _save(fig, out_dir, "activation_correlation_heatmaps")


def plot_influence_consistency(cc_consistency: dict, out_dir: str):
    """cc_consistency: {'raw_gradient': {layer: {...}}, 'cc_influence': {layer: {...}}} for the CC model only
    -- this is the metric's core question: within the SAME trained network, is the CC naturalized
    influence M_l a more batch-to-batch-stable signal than the raw Hebbian gradient magnitude?"""
    layer_names = list(cc_consistency["raw_gradient"].keys())
    raw_vals = [cc_consistency["raw_gradient"][l]["mean_consistency"] for l in layer_names]
    raw_err = [cc_consistency["raw_gradient"][l]["std_consistency"] for l in layer_names]
    inf_vals = [cc_consistency["cc_influence"][l]["mean_consistency"] for l in layer_names]
    inf_err = [cc_consistency["cc_influence"][l]["std_consistency"] for l in layer_names]

    x = np.arange(len(layer_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(x - width / 2, raw_vals, width, yerr=raw_err, label="raw |Hebbian dW| (backprop-style)",
           color="#888888", capsize=3)
    ax.bar(x + width / 2, inf_vals, width, yerr=inf_err, label=r"CC influence $M_l$",
           color=MODEL_COLORS["CC"], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names)
    ax.set_ylabel("mean batch-to-batch cosine consistency")
    ax.set_title("CC: stability of 'what matters', raw gradient vs naturalized influence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir, "influence_consistency_cc")


def plot_grad_consistency_overview(per_model_mean_consistency: dict, out_dir: str):
    """per_model_mean_consistency: model_name -> mean raw-gradient consistency averaged over its own layers."""
    fig, ax = plt.subplots(figsize=(5, 4))
    names = list(per_model_mean_consistency.keys())
    vals = [per_model_mean_consistency[n] for n in names]
    ax.bar(names, vals, color=[MODEL_COLORS.get(n) for n in names])
    ax.set_ylabel("mean batch-to-batch cosine consistency\nof raw gradient magnitude")
    ax.set_title("raw-gradient stability across models\n(context for the CC influence-consistency result)")
    fig.tight_layout()
    return _save(fig, out_dir, "raw_gradient_consistency_overview")


def plot_tsne(embeddings: dict, labels: dict, out_dir: str):
    n = len(embeddings)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6))
    if n == 1:
        axes = [axes]
    for ax, (name, emb) in zip(axes, embeddings.items()):
        y = labels[name]
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=8, alpha=0.8)
        ax.set_title(f"{name}: hidden-layer t-SNE, colored by class")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02, ticks=range(10))
    return _save(fig, out_dir, "tsne_representations")


def plot_summary_table(summary_rows: list, out_dir: str):
    """summary_rows: list of dicts with identical keys -> a saved-as-image table for a quick glance."""
    import pandas as pd
    df = pd.DataFrame(summary_rows).set_index("model")
    fig, ax = plt.subplots(figsize=(1.6 * len(df.columns) + 1, 0.6 * len(df) + 1.2))
    ax.axis("off")
    tbl = ax.table(cellText=np.round(df.values.astype(float), 4),
                    rowLabels=df.index, colLabels=df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    fig.tight_layout()
    return _save(fig, out_dir, "summary_table")
