"""Continual-learning metrics + CC-theory diagnostics.

CHANGE (v2): commutator_probe sets the trainer's active classes per task in
class-IL so probe updates use the same masked label error as real training
(otherwise the probe would measure interference that training no longer has).

Standard CL metrics from R[i, j] = accuracy on task j after training task i:
    ACC, BWT, FWT, Forget_j (see cl_summary).

CC-theory diagnostics:
  * leakage epsilon_c (locality-error proxy, shift note Sec. 3.2)
  * order-swap commutator proxy Delta_{c,c'} (CCL paper Sec. 2.4 / App. A),
    in parameter distance and in the paper's performance metric
    d_Theta = sup_c |L_c - L_c'|
  * support overlap (Jaccard) ~ confusion graph sparsity (CCL Thm 5)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


# ----------------------------------------------------------------------
def cl_summary(R: np.ndarray, rand_acc: float) -> Dict:
    T = R.shape[0]
    acc = float(R[T - 1].mean())
    bwt = float(np.mean([R[T - 1, j] - R[j, j] for j in range(T - 1)])) if T > 1 else 0.0
    fwt = float(np.mean([R[j - 1, j] - rand_acc for j in range(1, T)])) if T > 1 else 0.0
    forgetting = [float(max(R[: T, j].max() - R[T - 1, j], 0.0)) for j in range(T)]
    return {"ACC": acc, "BWT": bwt, "FWT": fwt,
            "forgetting_per_task": forgetting,
            "avg_forgetting": float(np.mean(forgetting[:-1])) if T > 1 else 0.0}


# ----------------------------------------------------------------------
def module_param_change(trainer, snap_before, snap_after, group_size: int):
    changes = []
    Wb, Wa = snap_before["W"], snap_after["W"]
    for k in range(len(Wa)):
        dW = (Wa[k] - Wb[k])
        n_child = dW.shape[0]
        row_norm = dW.norm(dim=1)
        for i in range(0, n_child, group_size):
            changes.append(float(row_norm[i:i + group_size].norm()))
    return torch.tensor(changes)


def leakage(trainer, snap_before, snap_after, support: torch.Tensor,
            group_size: int) -> Dict:
    ch = module_param_change(trainer, snap_before, snap_after, group_size)
    n = min(len(ch), len(support))
    ch, sup = ch[:n], support[:n]
    out = float(ch[~sup].sum())
    tot = float(ch.sum()) + 1e-12
    return {"eps_c": out, "eps_c_frac": out / tot,
            "module_change": ch.tolist(), "support": sup.tolist()}


# ----------------------------------------------------------------------
@torch.no_grad()
def _task_losses(trainer, eval_loaders, head_of):
    return [1.0 - trainer.evaluate(eval_loaders[t], head_of(t))
            for t in range(len(eval_loaders))]


def commutator_probe(trainer, probes, eval_loaders, head_of, n_batches: int,
                     data_module) -> Dict:
    """Delta_{i,j} via order swap from the CURRENT theta."""
    T = len(probes)
    base = trainer.snapshot()
    d_param = np.zeros((T, T))
    d_perf = np.zeros((T, T))
    class_il = getattr(data_module, "scenario", "") == "class_il"

    def batches_for(t):
        x, y = probes[t]
        bs = max(8, len(x) // n_batches)
        return [(x[i * bs:(i + 1) * bs], y[i * bs:(i + 1) * bs]) for i in range(n_batches)]

    def train_on(t):
        # NEW: same masked label error as real class-IL training
        if class_il and hasattr(trainer, "set_active_classes"):
            trainer.set_active_classes(data_module.task_classes[t])
        trainer.train_batches(batches_for(t), head_of(t))

    for i in range(T):
        for j in range(i + 1, T):
            trainer.load_snapshot(base)
            train_on(i); train_on(j)
            p_ij = trainer.flat_params().clone()
            L_ij = _task_losses(trainer, eval_loaders, head_of)

            trainer.load_snapshot(base)
            train_on(j); train_on(i)
            p_ji = trainer.flat_params().clone()
            L_ji = _task_losses(trainer, eval_loaders, head_of)

            d_param[i, j] = d_param[j, i] = float((p_ij - p_ji).norm())
            d_perf[i, j] = d_perf[j, i] = float(np.max(np.abs(np.array(L_ij) - np.array(L_ji))))
    trainer.load_snapshot(base)
    return {"param": d_param.tolist(), "perf": d_perf.tolist()}


# ----------------------------------------------------------------------
def support_overlap(supports: Dict[int, torch.Tensor]) -> Optional[np.ndarray]:
    if not supports:
        return None
    ts = sorted(supports)
    T = len(ts)
    J = np.zeros((T, T))
    for a in range(T):
        for b in range(T):
            sa, sb = supports[ts[a]].bool(), supports[ts[b]].bool()
            n = min(len(sa), len(sb))
            inter = float((sa[:n] & sb[:n]).sum())
            union = float((sa[:n] | sb[:n]).sum()) + 1e-12
            J[a, b] = inter / union
    return J
