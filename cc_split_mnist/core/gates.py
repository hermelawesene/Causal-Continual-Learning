"""Amplitude-preserving causal gate (pseudocode doc, step 3.3) + schedules.

Gate construction per layer l:

    R_l = NORMALIZE_INPUTWISE( |M_l|^p_t )        # per output row over inputs
    G_l = 1 + alpha_t * ( n_in * R_l - 1 )        # preserves update mass/row
    G_l = clamp( G_l,
                 min = 1 - alpha_t + floor*alpha_t,   # no dead-zero inputs
                 max = g_max )                        # NEW (v2): bound entries
    dW_l <- G_l * dW_l                            # gate the UPDATE only

WHY g_max (v2): the pseudocode preserves total update mass per output row but
does not bound individual entries. On wide layers (784 inputs) a peaked
influence distribution R makes single entries approach 1 + alpha*(n_in - 1),
i.e. hundreds of times the raw Hebbian step -> weight explosion -> NaN.
The CCL theory cares about the *relative* amplification between causal and
non-causal edges (selectivity), not the absolute multiplier, so a cap keeps
the semantics while restoring stability.

Schedules (pseudocode "Suggested schedules"):
    alpha_t : alpha_start -> alpha_end over the task's epochs
    p_t     : p_start -> p_end
    clarity : off for warmup epochs, then lambda_diff ramps to lambda_max
"""
from __future__ import annotations

import torch


def normalize_inputwise(T: torch.Tensor, eps: float = 1e-8):
    """Rows are outputs, columns inputs (W is (out, in)); normalize per row."""
    return T / (T.sum(dim=1, keepdim=True) + eps)


def build_gate(M: torch.Tensor, alpha: float, p: float, floor_frac: float,
               g_max: float = 3.0):
    R = normalize_inputwise(M.abs() ** p)
    n_in = M.shape[1]
    G = 1.0 + alpha * (n_in * R - 1.0)
    G = torch.clamp(G, min=1.0 - alpha + floor_frac * alpha, max=g_max)
    return G  # no autograd graph anywhere -> stop-grad by construction


class CCSchedule:
    """Per-(epoch, task) scalar schedule; linear ramps within each task."""

    def __init__(self, cc_cfg, epochs_per_task: int):
        g, c = cc_cfg["gate"], cc_cfg["clarity"]
        self.a0, self.a1 = g["alpha_start"], g["alpha_end"]
        self.p0, self.p1 = g["p_start"], g["p_end"]
        self.floor = g["floor_frac"]
        self.g_max = g.get("g_max", 3.0)
        self.warmup = c["warmup_epochs"]
        self.lam_max = c["lambda_max"] if c["enabled"] else 0.0
        self.E = max(1, epochs_per_task)

    def at(self, epoch: int):
        t = min(1.0, epoch / max(1, self.E - 1)) if self.E > 1 else 1.0
        alpha = self.a0 + t * (self.a1 - self.a0)
        p = self.p0 + t * (self.p1 - self.p0)
        if epoch < self.warmup or self.lam_max == 0.0:
            lam = 0.0
        else:
            ramp = (epoch - self.warmup + 1) / max(1, self.E - self.warmup)
            lam = self.lam_max * min(1.0, ramp)
        return {"alpha": alpha, "p": p, "lambda_diff": lam}
