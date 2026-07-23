"""Sequential continual-learning loop shared by all methods.

CHANGES (v2):
  * class-IL: sets the trainer's active classes before each task so the
    label error is masked to the current task's classes (PC and CC).
  * calls trainer.prepare_task(probe, t, head) before training when available
    (CC computes pre-training unit influence for the protection rule).
"""
from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np
import torch

from metrics.continual_metrics import (cl_summary, commutator_probe, leakage,
                                       support_overlap)
from plotting import plots


def run_continual(trainer, data, cfg, run_dir: str) -> Dict:
    T = data.num_tasks
    scenario = cfg["scenario"]
    head_of = (lambda t: t) if scenario == "task_il" else (lambda t: 0)
    rand_acc = 1.0 / data.label_dim()

    os.makedirs(run_dir, exist_ok=True)
    plot_dir = os.path.join(run_dir, "plots")

    eval_loaders = [data.test_loader(t) for t in range(T)]
    probes = [data.probe_batch(t, seed=cfg["seed"]) for t in range(T)]

    R = np.zeros((T, T))
    energy_traces: Dict[int, list] = {}
    leaks: Dict[int, Dict] = {}
    gsize = cfg["cc"]["influence"]["module_group_size"]

    def logger(task_id, ep, info):
        if "free_energy" in info:
            energy_traces.setdefault(task_id, []).append(info["free_energy"])
        msg = ", ".join(f"{k}={v:.4f}" for k, v in info.items())
        print(f"  [{trainer.name}] task {task_id} epoch {ep}: {msg}")

    for t in range(T):
        print(f"[{trainer.name}] === training task {t} ({scenario}) ===")

        # NEW: class-IL label-error masking
        if scenario == "class_il" and hasattr(trainer, "set_active_classes"):
            trainer.set_active_classes(data.task_classes[t])

        # NEW: pre-training influence for the protection rule (CC)
        if hasattr(trainer, "prepare_task"):
            trainer.prepare_task(probes[t], t, head_of(t))

        snap_before = trainer.snapshot()
        trainer.train_task(data.train_loader(t), head_of(t), t, logger=logger)

        if hasattr(trainer, "compute_support"):
            support, infl = trainer.compute_support(probes[t], t, head_of(t))
            leaks[t] = leakage(trainer, snap_before, trainer.snapshot(),
                               support, gsize)
            print(f"  support size={int(support.sum())}/{len(support)}, "
                  f"leakage frac={leaks[t]['eps_c_frac']:.3f}")

        for j in range(T):
            R[t, j] = trainer.evaluate(eval_loaders[j], head_of(j))
        print("  acc row:", np.round(R[t], 3).tolist())

    summary = cl_summary(R, rand_acc)
    print(f"[{trainer.name}] ACC={summary['ACC']:.3f} BWT={summary['BWT']:.3f} "
          f"FWT={summary['FWT']:.3f} avg_forget={summary['avg_forgetting']:.3f}")

    comm = commutator_probe(trainer, probes, eval_loaders, head_of,
                            cfg["metrics"]["commutator_batches"], data)

    supports = getattr(trainer, "supports", {})
    infl_by_task = {t: v.numpy() for t, v in
                    getattr(trainer, "influence_by_task", {}).items()}
    J = support_overlap(supports)

    plots.plot_accuracy_matrix(R, plot_dir, title=f"[{trainer.name}]")
    plots.plot_acc_over_time(R, plot_dir, title=f"[{trainer.name}]")
    plots.plot_forgetting(summary, plot_dir)
    plots.plot_commutator(comm, plot_dir)
    plots.plot_energy(energy_traces, plot_dir)
    plots.plot_influence_by_task(infl_by_task, plot_dir)
    plots.plot_supports({t: s.numpy() for t, s in supports.items()}, plot_dir)
    plots.plot_confusion_graph(J, plot_dir)
    plots.plot_leakage(leaks, plot_dir)
    if hasattr(trainer, "gate_snapshots"):
        plots.plot_gates(trainer.gate_snapshots, plot_dir)

    out = {"method": trainer.name, "scenario": scenario,
           "R": R.tolist(), "summary": summary,
           "commutator": comm,
           "leakage": leaks,
           "support_overlap": J.tolist() if J is not None else None}
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.save(os.path.join(run_dir, "acc_matrix.npy"), R)
    return out
