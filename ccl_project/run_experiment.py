"""
run_experiment.py  —  main entry point.

Usage:
    python run_experiment.py

All settings live in configs/config.py.
Outputs are written to ./outputs/.
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs       import CFG
from data          import get_split_mnist
from trainers      import NaiveTrainer, PCTrainer, CCTrainer
from visualization import plot_comparison, plot_gates, plot_per_task_accuracy


def banner(msg, char="═", w=68):
    b = char * w
    print(f"\n{b}\n  {msg}\n{b}")


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main():
    set_seed(CFG.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    banner(f"Causal Continual Learning — Split MNIST  [{device}]")
    print(f"  Mode       : {CFG.INCREMENTAL_MODE.upper()}-Incremental Learning")
    print(f"  Tasks      : {[f'{a}vs{b}' for a,b in CFG.TASKS]}")
    print(f"  Epochs/task: {CFG.N_EPOCHS}")
    print(f"  Hidden     : {CFG.HIDDEN_DIM}  ({CFG.N_MODULES} modules × {CFG.MODULE_DIM} each)")
    print(f"  Clarity λ  : {CFG.CLARITY_W}")
    print(f"  Gate freq  : every {CFG.GATE_FREQ} steps")
    print(f"  Outputs    : {CFG.OUTPUT_DIR}/")

    print("\n  Loading Split MNIST ...")
    train_loaders, test_loaders, task_names = get_split_mnist()

    # ── METHOD 1: Naive MLP ───────────────────────────────────────
    banner("METHOD 1 — Naive MLP  (standard backprop, no protection)")
    set_seed(CFG.SEED)
    naive_t = NaiveTrainer(train_loaders, test_loaders, task_names, device)
    res_naive = naive_t.run(verbose=True)
    res_naive.print_summary()

    # ── METHOD 2: Discriminative PC Network ───────────────────────
    banner("METHOD 2 — PC Network  (discriminative PC, modular, no gates)")
    set_seed(CFG.SEED)
    pc_t = PCTrainer(train_loaders, test_loaders, task_names, device)
    res_pc = pc_t.run(verbose=True)
    res_pc.print_summary()

    # ── METHOD 3: PC + Causal Coding ─────────────────────────────
    banner("METHOD 3 — PC + Causal Coding  (Jacobian gates + clarity penalty)")
    set_seed(CFG.SEED)
    cc_t = CCTrainer(train_loaders, test_loaders, task_names, device)
    res_cc = cc_t.run(verbose=True)
    res_cc.print_summary()

    # ── Final table ───────────────────────────────────────────────
    banner("RESULTS SUMMARY")
    hdr = f"  {'Method':<24}  {'BWT':>8}  {'Task-1 drop':>12}  {'Final avg':>10}"
    print(hdr); print("-" * len(hdr))
    for res in [res_naive, res_pc, res_cc]:
        print(f"  {res.name:<24}  "
              f"{res.backward_transfer():>+7.1f}%  "
              f"{res.forgetting(0):>+11.1f}%  "
              f"{res.final_avg_accuracy():>9.1f}%")

    # ── Plots ─────────────────────────────────────────────────────
    banner("GENERATING PLOTS")

    print("  [1/3] Main comparison figure ...")
    plot_comparison([res_naive, res_pc, res_cc], task_names,
                    output_dir=CFG.OUTPUT_DIR,
                    save=CFG.SAVE_FIG, show=CFG.SHOW_FIG)

    print("  [2/3] Per-task accuracy figure ...")
    plot_per_task_accuracy([res_naive, res_pc, res_cc], task_names,
                            output_dir=CFG.OUTPUT_DIR,
                            save=CFG.SAVE_FIG, show=CFG.SHOW_FIG)

    print("  [3/3] Causal gates figure ...")
    if cc_t.gate_history_l1:
        plot_gates(cc_t.gate_history_l1, cc_t.gate_history_l2, task_names,
                   output_dir=CFG.OUTPUT_DIR,
                   save=CFG.SAVE_FIG, show=CFG.SHOW_FIG)
    else:
        print("       (no gate history — skipped)")

    banner("DONE")
    print(f"  Outputs saved to → {CFG.OUTPUT_DIR}/")
    for f in ["comparison.png", "per_task_accuracy.png", "causal_gates.png"]:
        full = os.path.join(CFG.OUTPUT_DIR, f)
        exists = "✓" if os.path.exists(full) else "✗"
        print(f"    {exists}  {f}")


if __name__ == "__main__":
    main()