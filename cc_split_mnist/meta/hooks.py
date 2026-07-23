"""PHASE-2 HOOKS (distribution-shift / meta-learning note, Dec 2025).

* InfluenceTracker — running (mu, sigma) per module influence (note Sec. 2.3)
* lcb_gate         — confidence-aware gating r = psi(LCB - tau) with
                     eps-exploration mixing UCB (note Sec. 2.5, 3.1)
* DriftDetector    — short-vs-long window D_t = max_k |I_short - I_long|
                     for mechanism-shift segmentation (note Sec. 2.4)

Wired into CCTrainer behind cc.meta.enabled (default: false).
"""
from __future__ import annotations

from collections import deque

import torch


class InfluenceTracker:
    def __init__(self, n_modules: int, momentum: float = 0.9, device="cpu"):
        self.mu = torch.zeros(n_modules, device=device)
        self.var = torch.ones(n_modules, device=device)
        self.m = momentum
        self.count = 0

    def update(self, infl: torch.Tensor):
        if self.count == 0:
            self.mu = infl.clone()
            self.var = torch.ones_like(infl) * infl.var().clamp_min(1e-8)
        else:
            delta = infl - self.mu
            self.mu = self.m * self.mu + (1 - self.m) * infl
            self.var = self.m * self.var + (1 - self.m) * delta ** 2
        self.count += 1

    def sigma(self):
        return self.var.clamp_min(1e-12).sqrt()


def lcb_gate(mu, sigma, tau, beta=1.0, explore_eps=0.05, temp=10.0):
    lcb = mu - beta * sigma
    ucb = mu + beta * sigma
    psi = lambda v: torch.sigmoid(temp * (v - tau))
    return (1 - explore_eps) * psi(lcb) + explore_eps * psi(ucb)


class DriftDetector:
    def __init__(self, short: int = 20, long: int = 200):
        self.short = deque(maxlen=short)
        self.long = deque(maxlen=long)

    def update(self, infl: torch.Tensor) -> float:
        self.short.append(infl.clone())
        self.long.append(infl.clone())
        if len(self.long) < 2:   # v2: we update once per task/refresh
            return 0.0
        s = torch.stack(list(self.short)).mean(0)
        l = torch.stack(list(self.long)).mean(0)
        return float((s - l).abs().max())
