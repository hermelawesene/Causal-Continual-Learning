"""
run_experiment.py
==================
Trains MLP, discriminative-PC, and discriminative-CC on the same data
under the same epoch budget, then runs every metric in metrics.py and
every plot in plotting.py, saving everything under --output_dir.

Usage:
    python run_experiment.py --smoke                 # ~30s pipeline sanity check (sklearn `digits`)
    python run_experiment.py                          # full MNIST run (needs internet or a local cache)
    python run_experiment.py --epochs 20 --seed 1
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from config import make_smoke_configs, make_full_configs
from data import load_splits
from models import MLP, PCNetwork, CCNetwork
from schedule import CCSchedule
from utils import set_seed, Batcher, accuracy, ensure_dir
import metrics as M
import plotting as P


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny/fast pipeline check on sklearn digits")
    ap.add_argument("--dataset", type=str, default=None, choices=[None, "mnist", "digits"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    return ap.parse_args()


def get_first_hidden_activation(model_name, model, X):
    """The 'first hidden layer away from the input' for each model family,
    so the entanglement/representation metrics are computed at a comparable
    depth across MLP / PC / CC."""
    if model_name == "MLP":
        return model.get_hidden_activations(X, layer_idx=1)
    else:
        return model.get_hidden_activations(X, layer_idx=model.L - 1)


def collect_raw_grad_sequence(model_name, model, batcher, n_batches):
    seqs = {}
    count = 0
    for xb, yb in batcher:
        if model_name == "MLP":
            g = model.compute_grads_no_update(xb, yb)
        else:
            out = model.compute_grads_no_update(xb, yb, T=1.0)
            g = {k: v[0] for k, v in out["grads"].items()}
        for k, v in g.items():
            seqs.setdefault(k, []).append(v)
        count += 1
        if count >= n_batches:
            break
    return seqs


def collect_cc_M_sequence(cc_model, batcher, n_batches):
    grad_seqs, M_seqs = {}, {}
    count = 0
    for xb, yb in batcher:
        out = cc_model.compute_M_no_update(xb, yb, T=1.0)
        for k, (gW, gb) in out["grads"].items():
            grad_seqs.setdefault(k, []).append(gW)
        for k, Mv in out["M"].items():
            M_seqs.setdefault(k, []).append(Mv)
        count += 1
        if count >= n_batches:
            break
    return grad_seqs, M_seqs


def train_mlp(model, Xtr, ytr, Xval, yval, data_cfg, n_epochs, seed):
    for epoch in range(n_epochs):
        batcher = Batcher(Xtr, ytr, data_cfg.batch_size, seed=seed * 1000 + epoch)
        losses = [model.train_step(xb, yb) for xb, yb in batcher]
        probs = model.predict(Xval)
        model.history["train_loss"].append(float(np.mean(losses)))
        model.history["val_loss"].append(M.performance_metrics(probs, yval, model.sizes[-1])["loss"])
        model.history["val_acc"].append(accuracy(probs, yval))
        print(f"  [MLP] epoch {epoch+1}/{n_epochs}  train_loss={model.history['train_loss'][-1]:.4f}  "
              f"val_acc={model.history['val_acc'][-1]:.4f}")


def train_pc(model, Xtr, ytr, Xval, yval, data_cfg, sched, n_epochs, seed):
    for epoch in range(n_epochs):
        T_t = sched.get(epoch)["T_t"]
        batcher = Batcher(Xtr, ytr, data_cfg.batch_size, seed=seed * 1000 + epoch)
        losses = [model.train_step(xb, yb, T=T_t) for xb, yb in batcher]
        probs = model.predict(Xval, T=1.0)
        model.history["train_loss"].append(float(np.mean(losses)))
        model.history["val_loss"].append(M.performance_metrics(probs, yval, model.n_classes)["loss"])
        model.history["val_acc"].append(accuracy(probs, yval))
        print(f"  [PC]  epoch {epoch+1}/{n_epochs}  train_loss={model.history['train_loss'][-1]:.4f}  "
              f"val_acc={model.history['val_acc'][-1]:.4f}")


def train_cc(model, Xtr, ytr, Xval, yval, data_cfg, sched, n_epochs, seed):
    for epoch in range(n_epochs):
        st = sched.get(epoch)
        batcher = Batcher(Xtr, ytr, data_cfg.batch_size, seed=seed * 1000 + epoch)
        losses, gate_means, gate_stds, gate_supp = [], [], [], []
        for xb, yb in batcher:
            loss, info = model.train_step(xb, yb, st)
            losses.append(loss)
            gate_means.append(info["gate_mean"])
            gate_stds.append(info["gate_std"])
            gate_supp.append(info["gate_suppressed_frac"])
        probs = model.predict(Xval, T=1.0)
        model.history["train_loss"].append(float(np.mean(losses)))
        model.history["val_loss"].append(M.performance_metrics(probs, yval, model.n_classes)["loss"])
        model.history["val_acc"].append(accuracy(probs, yval))
        model.history["gate_mean"].append(float(np.mean(gate_means)))
        model.history["gate_std"].append(float(np.mean(gate_stds)))
        model.history["gate_suppressed_frac"].append(float(np.mean(gate_supp)))
        model.history["alpha_t"].append(st["alpha_t"])
        model.history["p_t"].append(st["p_t"])
        model.history["lambda_diff_t"].append(st["lambda_diff_t"])
        model.history["clarity_on"].append(st["clarity_on"])
        print(f"  [CC]  epoch {epoch+1}/{n_epochs} [{st['phase']:>9s}]  "
              f"train_loss={model.history['train_loss'][-1]:.4f}  "
              f"val_acc={model.history['val_acc'][-1]:.4f}  gate_mean={model.history['gate_mean'][-1]:.4f}  "
              f"clarity={st['clarity_on']}")


def main():
    args = parse_args()
    if args.smoke:
        data_cfg, arch_cfg, infer_cfg, sched_cfg, optim_cfg, exp_cfg = make_smoke_configs()
    else:
        data_cfg, arch_cfg, infer_cfg, sched_cfg, optim_cfg, exp_cfg = make_full_configs()

    if args.dataset:
        data_cfg.dataset = args.dataset
    if args.epochs:
        sched_cfg.total_epochs = args.epochs
        sched_cfg.warmup_epochs = max(1, args.epochs // 4)
        sched_cfg.soft_gate_epochs = max(1, args.epochs // 4)
        sched_cfg.clarity_warmup_epochs = max(1, args.epochs // 6)
    if args.seed is not None:
        exp_cfg.seed = args.seed
    if args.output_dir:
        exp_cfg.output_dir = args.output_dir

    set_seed(exp_cfg.seed)
    out_dir = ensure_dir(exp_cfg.output_dir)
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))

    print(f"=== Experiment: {exp_cfg.name} | dataset={data_cfg.dataset} | "
          f"epochs={sched_cfg.total_epochs} | seed={exp_cfg.seed} ===")

    splits = load_splits(data_cfg, seed=exp_cfg.seed)
    Xtr, ytr = splits["train"]
    Xval, yval = splits["val"]
    Xte, yte = splits["test"]
    n_classes, input_dim = splits["n_classes"], splits["input_dim"]
    print(f"train={Xtr.shape}, val={Xval.shape}, test={Xte.shape}, "
          f"n_classes={n_classes}, real_mnist={splits['is_real_mnist']}")

    n_epochs = sched_cfg.total_epochs
    sched = CCSchedule(sched_cfg, total_epochs=n_epochs)

    mlp = MLP(arch_cfg, n_classes, input_dim, optim_cfg, seed=exp_cfg.seed + 1)
    pc = PCNetwork(arch_cfg, infer_cfg, n_classes, input_dim, optim_cfg, seed=exp_cfg.seed + 2)
    cc = CCNetwork(arch_cfg, infer_cfg, n_classes, input_dim, optim_cfg, seed=exp_cfg.seed + 3)

    t0 = time.time()
    print("\n--- training MLP (control) ---")
    train_mlp(mlp, Xtr, ytr, Xval, yval, data_cfg, n_epochs, exp_cfg.seed)
    print("\n--- training discriminative PC (no gating, no clarity) ---")
    train_pc(pc, Xtr, ytr, Xval, yval, data_cfg, sched, n_epochs, exp_cfg.seed)
    print("\n--- training discriminative CC (warmup -> soft-gate -> full CC) ---")
    train_cc(cc, Xtr, ytr, Xval, yval, data_cfg, sched, n_epochs, exp_cfg.seed)
    print(f"\ntotal training time: {time.time() - t0:.1f}s")

    models = {"MLP": mlp, "PC": pc, "CC": cc}

    # ----------------------------- (1) performance ----------------------------- #
    print("\n--- evaluating on test set ---")
    perf = {}
    for name, model in models.items():
        probs = model.predict(Xte) if name == "MLP" else model.predict(Xte, T=1.0)
        perf[name] = M.performance_metrics(probs, yte, n_classes)
        print(f"  {name}: test_acc={perf[name]['accuracy']:.4f}  test_loss={perf[name]['loss']:.4f}  "
              f"ece={perf[name]['ece']:.4f}")

    # ----------------------------- (2) sparsity / modularity ----------------------------- #
    sparsity = {name: M.sparsity_report(model.get_weight_matrices()) for name, model in models.items()}

    # ----------------------------- (3) entanglement ----------------------------- #
    entangle = {}
    for name, model in models.items():
        acts = get_first_hidden_activation(name, model, Xte)
        entangle[name] = M.entanglement_report(acts, yte, n_classes)

    # ----------------------------- (4) influence consistency ----------------------------- #
    n_cons_batches = exp_cfg.influence_consistency_batches
    raw_overview = {}
    cc_consistency = None
    for name, model in models.items():
        batcher = Batcher(Xtr, ytr, data_cfg.batch_size, shuffle=True, seed=exp_cfg.seed + 99)
        if name == "CC":
            grad_seqs, M_seqs = collect_cc_M_sequence(model, batcher, n_cons_batches)
            cc_consistency = M.influence_consistency_report(grad_seqs, M_seqs)
            raw_overview[name] = float(np.mean([v["mean_consistency"] for v in cc_consistency["raw_gradient"].values()]))
        else:
            grad_seqs = collect_raw_grad_sequence(name, model, batcher, n_cons_batches)
            rep = M.influence_consistency_report(grad_seqs, None)
            raw_overview[name] = float(np.mean([v["mean_consistency"] for v in rep["raw_gradient"].values()]))

    # ----------------------------- (5) representation structure ----------------------------- #
    print("\n--- computing t-SNE embeddings ---")
    embeddings, labels_for_tsne, silhouettes = {}, {}, {}
    for name, model in models.items():
        acts = get_first_hidden_activation(name, model, Xte)
        n = acts.shape[0]
        idx = np.random.RandomState(0).choice(n, min(exp_cfg.tsne_n_samples, n), replace=False)
        emb = M.tsne_embedding(acts[idx], n_samples=exp_cfg.tsne_n_samples, seed=exp_cfg.seed)
        embeddings[name] = emb
        labels_for_tsne[name] = yte[idx]
        silhouettes[name] = M.silhouette_of_classes(emb, yte[idx])
        print(f"  {name}: tsne silhouette={silhouettes[name]:.4f}")

    # =============================== PLOTTING =============================== #
    print("\n--- saving figures ---")
    histories = {name: model.history for name, model in models.items()}
    P.plot_training_curves(histories, fig_dir, "val_acc", "validation accuracy", "val_accuracy_curve")
    P.plot_training_curves(histories, fig_dir, "train_loss", "training loss", "train_loss_curve", logy=True)
    P.plot_training_curves(histories, fig_dir, "val_loss", "validation loss", "val_loss_curve", logy=True)
    P.plot_cc_phase_diagnostics(cc.history, fig_dir)
    P.plot_confusion_matrices({name: perf[name]["confusion_matrix"] for name in models}, fig_dir)
    P.plot_weight_heatmaps({"MLP": mlp.get_weight_matrices(), "PC": pc.get_weight_matrices(),
                             "CC": cc.get_weight_matrices()}, fig_dir, layer_idx=0)
    P.plot_sparsity_bars(sparsity, fig_dir)
    P.plot_effective_rank(sparsity, fig_dir)
    P.plot_entanglement_bars(entangle, fig_dir)
    P.plot_entanglement_histograms(entangle, fig_dir)
    P.plot_correlation_heatmaps(entangle, fig_dir)
    if cc_consistency is not None:
        P.plot_influence_consistency(cc_consistency, fig_dir)
    P.plot_grad_consistency_overview(raw_overview, fig_dir)
    P.plot_tsne(embeddings, labels_for_tsne, fig_dir)

    summary_rows = []
    for name in models:
        summary_rows.append(dict(
            model=name,
            test_acc=perf[name]["accuracy"],
            test_loss=perf[name]["loss"],
            ece=perf[name]["ece"],
            overall_sparsity=sparsity[name]["overall_sparsity"],
            mean_entanglement=entangle[name]["mean_entanglement"],
            mean_neuron_label_mi=entangle[name]["mean_neuron_label_mi"],
            mean_abs_pairwise_corr=entangle[name]["mean_abs_pairwise_corr"],
            tsne_silhouette=silhouettes[name],
            raw_grad_consistency=raw_overview[name],
        ))
    P.plot_summary_table(summary_rows, fig_dir)

    # =============================== SAVE JSON REPORT =============================== #
    report = dict(
        config=dict(dataset=data_cfg.dataset, epochs=n_epochs, seed=exp_cfg.seed,
                     sizes=arch_cfg.sizes, is_real_mnist=splits["is_real_mnist"]),
        summary=summary_rows,
        cc_final_gamma=cc.gamma,
        pc_final_gamma=pc.gamma,
    )
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    print("\n=== SUMMARY ===")
    for row in summary_rows:
        print(f"  {row['model']:>4s}: test_acc={row['test_acc']:.4f}  ece={row['ece']:.4f}  "
              f"sparsity={row['overall_sparsity']:.4f}  entanglement={row['mean_entanglement']:.3f}  "
              f"label_MI={row['mean_neuron_label_mi']:.4f}  raw_grad_consistency={row['raw_grad_consistency']:.4f}")
    print(f"\nAll figures + report.json saved under: {out_dir}")


if __name__ == "__main__":
    main()
