"""Trainers for vanilla discriminative PC and Causal Coding (CC).

CHANGES (v2):
  * set_active_classes() + label-error masking (class-IL): the label error
    only touches the current task's classes, so old/future output weights
    are no longer actively destroyed. Applied to BOTH PC and CC so the
    comparison stays fair; the remaining forgetting is trunk feature drift,
    which is what CC's machinery is supposed to reduce.
  * TASK-CONDITIONED PROTECTION (the missing half of CC-for-CL): the neural
    analogue of the CCL paper's "freeze outside support" (Sec. 8.3, item 4).
    We track Omega_l(u) = max over PAST tasks of unit u's normalized
    do-influence on the label (Schur-Fisher composed C^{l->L}). While
    training task c, rows (child units) with high Omega but low CURRENT
    influence get their updates suppressed:
        prot(u) = 1 / (1 + strength * relu(Omega(u) - now(u)))
    EWC-shaped, but importance = do-influence instead of the Fisher — which
    is precisely the CC claim. Amplification (gates) + protection together
    give the locality the CCL theorem needs; amplification alone (v1) can't
    reduce forgetting, as your leakage ~0.6 numbers showed.
  * prepare_task(): computes current-task unit influence BEFORE training
    (needed by the protection rule; also matches the paper's "estimate
    influences -> form support -> then update" ordering).
  * META WIRING (phase 2, cc.meta.enabled=true): InfluenceTracker feeds an
    LCB/UCB confidence-aware gate per module (shift note Sec. 2.5/3.1),
    expanded to per-unit row multipliers; DriftDetector prints D_t at every
    support refresh (Sec. 2.4) — task boundaries are mechanism shifts, so
    you should see it spike there.
  * g_max passed through to build_gate (NaN fix).
  * CC-specific lr override (cc.lr_weights).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from core.clarity import clarity_grads
from core.gates import CCSchedule, build_gate
from core.influence import composed_influence, weight_proxy_M
from models.pc_network import DiscriminativePCNet
from meta.hooks import DriftDetector, InfluenceTracker, lcb_gate


def onehot(y, n, device):
    return F.one_hot(y, n).float().to(device)


class PCTrainer:
    name = "pc"

    def __init__(self, cfg, label_dim, n_heads, device):
        self.cfg = cfg
        self.device = device
        self.net = DiscriminativePCNet(cfg, label_dim, n_heads, device)
        self.p = cfg["pc"]
        self.epochs = self.p["epochs_per_task"]
        self.label_dim = label_dim
        self.active_mask: Optional[torch.Tensor] = None      # NEW
        self._batch_counter = 0

    # NEW: class-IL label-error masking ---------------------------------
    def set_active_classes(self, classes):
        m = torch.zeros(1, self.label_dim, device=self.device)
        m[0, list(classes)] = 1.0
        self.active_mask = m

    # ------------------------------------------------------------------
    def _step(self, x, y, head_id, epoch):
        state = self.net.infer(x, head_id,
                               y_onehot=onehot(y, self.label_dim, self.device),
                               label_mask=self.active_mask)
        deltas, pres = self.net.hebbian_deltas(state, head_id)
        gates = self._gates(state, pres, head_id, epoch, deltas)
        self.net.apply_deltas(deltas, head_id, self.p["lr_weights"],
                              self.p["weight_l2"], self.p["weight_l1"], gates=gates)
        if self.p["learn_precision"]:
            self.net.precision_step(state, head_id, self.p["lr_precision"])
        lat = self.p["lateral"]
        if lat["mode"] != "none" and lat.get("learn", False) \
                and self._batch_counter % lat["update_every"] == 0:
            self.net.lateral_step(state, head_id, lat["lr"])
        self._batch_counter += 1
        return state

    def _gates(self, state, pres, head_id, epoch, deltas):
        return None  # vanilla PC: no gates

    # ------------------------------------------------------------------
    def train_task(self, loader, head_id, task_id, logger=None):
        for ep in range(self.epochs):
            energy = 0.0
            n = 0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                state = self._step(x, y, head_id, ep)
                with torch.no_grad():
                    e = sum(((state["z"][k + 1] - state["mu"][k]) ** 2).mean()
                            for k in range(self.net.L))
                energy += float(e)
                n += 1
            if logger:
                logger(task_id, ep, {"free_energy": energy / max(1, n)})

    def train_batches(self, batches, head_id):
        for x, y in batches:
            self._step(x.to(self.device), y.to(self.device), head_id, epoch=0)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader, head_id):
        correct = total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.net.predict(x, head_id)
            correct += int((pred == y).sum())
            total += len(y)
        return correct / max(1, total)

    def flat_params(self):
        return self.net.flat_params()

    def snapshot(self):
        return self.net.snapshot()

    def load_snapshot(self, s):
        self.net.load_snapshot(s)

    # ---- influence utilities (shared with CC) --------------------------
    @torch.no_grad()
    def _unit_influence(self, probe, head_id) -> Dict[int, torch.Tensor]:
        """Per-unit normalized do-influence on the label, per hidden layer."""
        x, _ = probe
        state = self.net.infer(x.to(self.device), head_id, y_onehot=None)
        out = composed_influence(self.net, state, head_id,
                                 ridge=self.cfg["cc"]["influence"]["ridge"])
        return {l: (s / (s.max() + 1e-12)).cpu()
                for l, s in out["unit_scores"].items()}

    def _units_to_modules(self, unit_scores: Dict[int, torch.Tensor]) -> torch.Tensor:
        gsize = self.cfg["cc"]["influence"]["module_group_size"]
        mods = []
        for l in sorted(unit_scores):
            s = unit_scores[l]
            for i in range(0, s.shape[0], gsize):
                mods.append(float(s[i:i + gsize].mean()))
        return torch.tensor(mods)

    @torch.no_grad()
    def module_influence(self, probe, head_id):
        return self._units_to_modules(self._unit_influence(probe, head_id))


class CCTrainer(PCTrainer):
    name = "cc"

    def __init__(self, cfg, label_dim, n_heads, device):
        super().__init__(cfg, label_dim, n_heads, device)
        self.cc = cfg["cc"]
        if self.cc.get("lr_weights"):                        # NEW: CC lr override
            self.p["lr_weights"] = self.cc["lr_weights"]
        self.sched = CCSchedule(self.cc, self.p["epochs_per_task"])
        self.lam_prot = float(self.cc.get("protect_strength", 8.0))   # NEW
        self.supports: Dict[int, torch.Tensor] = {}
        self.influence_by_task: Dict[int, torch.Tensor] = {}
        self.gate_snapshots: Dict[int, List[torch.Tensor]] = {}
        # protection state (NEW)
        self.unit_scores_past: Optional[Dict[int, torch.Tensor]] = None  # Omega
        self._unit_scores_now: Optional[Dict[int, torch.Tensor]] = None
        # phase-2 meta (NOW WIRED)
        self.meta_on = self.cc["meta"]["enabled"]
        self._tracker: Optional[InfluenceTracker] = None
        self._drift = DriftDetector(self.cc["meta"]["drift_short_window"],
                                    self.cc["meta"]["drift_long_window"])
        self._meta_unit_r: Optional[Dict[int, torch.Tensor]] = None

    # NEW: called by the continual loop BEFORE training each task ---------
    def prepare_task(self, probe, task_id, head_id):
        self._unit_scores_now = self._unit_influence(probe, head_id)

    # ------------------------------------------------------------------
    def _gates(self, state, pres, head_id, epoch, deltas):
        sch = self.sched.at(epoch)
        Ms = weight_proxy_M(self.net, state, pres, head_id)
        gates = [build_gate(M, sch["alpha"], sch["p"], self.sched.floor,
                            g_max=self.sched.g_max) for M in Ms]

        # NEW: task-conditioned protection on trunk rows (child units) -----
        if self.unit_scores_past is not None:
            for k in range(self.net.L - 1):                  # child layer = k+1
                l = k + 1
                omega = self.unit_scores_past[l].to(gates[k].device)
                now = (self._unit_scores_now[l].to(gates[k].device)
                       if self._unit_scores_now is not None
                       else torch.zeros_like(omega))
                prot = 1.0 / (1.0 + self.lam_prot * torch.relu(omega - now))
                gates[k] = gates[k] * prot.unsqueeze(1)

        # NEW: meta LCB gate multipliers (per-unit rows), phase 2 ----------
        if self.meta_on and self._meta_unit_r is not None:
            for k in range(self.net.L - 1):
                r = self._meta_unit_r.get(k + 1)
                if r is not None:
                    gates[k] = gates[k] * r.to(gates[k].device).unsqueeze(1)

        # clarity: shrink edges overshadowed by diffusive multi-hop routes
        if sch["lambda_diff"] > 0:
            Ds = clarity_grads(self.net, Ms, head_id,
                               order=self.cc["clarity"]["diffusion_order"],
                               delta=self.cc["clarity"]["delta"])
            for k in range(self.net.L):
                dW, db = deltas[k]
                deltas[k] = (dW - sch["lambda_diff"] * Ds[k], db)

        self._last_gates = [g.detach().cpu() for g in gates]
        return gates

    # ------------------------------------------------------------------
    def _modules_to_units(self, mods: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Expand per-module values back to per-unit vectors per hidden layer."""
        gsize = self.cfg["cc"]["influence"]["module_group_size"]
        out, i = {}, 0
        for l in range(1, self.net.L):
            n = self.net.dims[l]
            vals = []
            for s in range(0, n, gsize):
                width = min(gsize, n - s)
                vals.append(mods[i].repeat(width))
                i += 1
            out[l] = torch.cat(vals)
        return out

    def compute_support(self, probe, task_id, head_id):
        """Refresh exact composed influence AFTER training a task; derive the
        module support S_c; update the protection memory Omega; feed meta."""
        unit_now = self._unit_influence(probe, head_id)
        infl = self._units_to_modules(unit_now)
        q = self.cfg["cc"]["influence"]["support_tau_quantile"]
        tau = torch.quantile(infl, q)
        support = infl >= tau
        self.supports[task_id] = support
        self.influence_by_task[task_id] = infl
        if hasattr(self, "_last_gates"):
            self.gate_snapshots[task_id] = self._last_gates

        # NEW: update Omega = max over past tasks of per-unit influence
        self._unit_scores_now = unit_now
        if self.unit_scores_past is None:
            self.unit_scores_past = {l: v.clone() for l, v in unit_now.items()}
        else:
            self.unit_scores_past = {l: torch.maximum(self.unit_scores_past[l],
                                                      unit_now[l])
                                     for l in unit_now}

        # NEW: meta wiring (LCB gate + drift statistic)
        if self.meta_on:
            if self._tracker is None:
                self._tracker = InfluenceTracker(len(infl))
            self._tracker.update(infl)
            d = self._drift.update(infl)
            tau_m = torch.quantile(self._tracker.mu, q)
            r_mod = lcb_gate(self._tracker.mu, self._tracker.sigma(), tau_m,
                             beta=self.cc["meta"]["lcb_beta"],
                             explore_eps=self.cc["meta"]["explore_eps"])
            # normalize to max 1: keeps confidence-aware SELECTIVITY between
            # modules without globally shrinking the effective learning rate
            r_mod = r_mod / (r_mod.max() + 1e-12)
            self._meta_unit_r = self._modules_to_units(r_mod)
            print(f"  [meta] drift D_t={d:.4f}, "
                  f"LCB gate mean={float(r_mod.mean()):.3f}")
        return support, infl
