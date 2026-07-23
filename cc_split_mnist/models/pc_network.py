"""Discriminative Predictive Coding network with lateral couplings and
learnable diagonal precisions.

CHANGES (v2):
  * label_head option: "gaussian" (classic; z^L latent clamped to one-hot)
    or "ce" (softmax/cross-entropy head; label layer = observed logits,
    eps_L = (onehot - softmax(mu_L / T)) / T, the discriminative analogue of
    the pseudocode's "crisp dL/dlogit gradient with temperature").
  * label_mask support in infer(): zeroes label-layer error entries for
    classes NOT in the current task. In class-IL this stops the head from
    actively pushing DOWN the outputs of absent (old/future) classes on
    every batch — which is what produced your exact-0.0 accuracy rows.
  * effective_label_pi(): categorical Fisher diagonal p(1-p)/T for the CE
    head, used by core/influence.py as the label-layer curvature block
    (the natural Gauss-Newton object the influence note is built on).

Mapping generative -> discriminative and all other derivations are
unchanged from v1; see README Sec. 1-2.

    F({z}) = sum_{k=1..L} 1/2 || z^k - g_{k-1}(z^{k-1}) ||^2_{Pi_k}
           + sum_{k=1..L} 1/2 (z^k)^T Lambda_k z^k  (+ L1, - 1/2 log det terms)

Inference:  dF/dz^k = eps^k + Lambda_k z^k - J_k^T eps^{k+1}
Learning:   Delta W_k ∝ eps^{k+1} phi(z^k)^T      (local, Hebbian)

Sign convention: eps is always "target-ish minus prediction" scaled by the
layer curvature, so W += lr * eps pre^T is DESCENT for both heads:
  gaussian: eps^L = Pi_L (z^L - mu^L), z^L clamped to onehot in training
  ce:       eps^L = (onehot - softmax(mu^L/T)) / T   (= -dCE/dlogits)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
def act_fns(name: str):
    if name == "tanh":
        return torch.tanh, lambda a: 1.0 - torch.tanh(a) ** 2
    if name == "sigmoid":
        return torch.sigmoid, lambda a: torch.sigmoid(a) * (1 - torch.sigmoid(a))
    if name == "relu":
        return torch.relu, lambda a: (a > 0).float()
    raise ValueError(name)


class Lateral:
    """Lambda = eps*I + B B^T, PSD low-rank lateral coupling for one layer."""

    def __init__(self, dim, rank, init_scale, eps_ridge, device):
        self.B = init_scale * torch.randn(dim, rank, device=device)
        self.eps = eps_ridge
        self.dim = dim

    def matvec(self, Z):
        return self.eps * Z + (Z @ self.B) @ self.B.T

    def full(self):
        return self.eps * torch.eye(self.dim, device=self.B.device) + self.B @ self.B.T

    def learn_step(self, Z, lr):
        """Descend F = 1/2 E[z^T Lambda z] - 1/2 log det Lambda w.r.t. B."""
        cov = (Z.T @ Z) / max(1, Z.shape[0])
        inv = torch.linalg.inv(self.full())
        grad = (cov - inv) @ self.B
        self.B -= lr * grad
        with torch.no_grad():
            self.B.clamp_(-5.0, 5.0)


@dataclass
class Head:
    W: torch.Tensor
    b: torch.Tensor
    log_pi: torch.Tensor
    lateral: Optional[Lateral]


class DiscriminativePCNet:
    def __init__(self, cfg, label_dim: int, n_heads: int, device):
        m, p = cfg["model"], cfg["pc"]
        self.cfg = cfg
        self.device = device
        self.dims = [m["input_dim"]] + list(m["hidden_dims"]) + [label_dim]
        self.L = len(self.dims) - 1
        self.phi, self.dphi = act_fns(m["activation"])
        self.label_head = p.get("label_head", "gaussian")     # NEW
        self.ce_temp = float(p.get("ce_temperature", 1.0))    # NEW

        def w_init(i, o):
            return torch.randn(o, i, device=device) * math.sqrt(1.0 / i)

        self.W = [w_init(self.dims[k], self.dims[k + 1]) for k in range(self.L - 1)]
        self.b = [torch.zeros(self.dims[k + 1], device=device) for k in range(self.L - 1)]
        self.log_pi = [torch.full((self.dims[k],), float(p["log_pi_init"]), device=device)
                       for k in range(1, self.L)]

        lat = p["lateral"]
        self.lat_mode = lat["mode"]
        self.laterals: List[Optional[Lateral]] = []
        for k in range(1, self.L):
            self.laterals.append(
                Lateral(self.dims[k], lat["rank"], lat["init_scale"], lat["eps_ridge"], device)
                if self.lat_mode == "lowrank" else None)

        self.heads: List[Head] = []
        for _ in range(n_heads):
            use_lat = (self.lat_mode == "lowrank" and lat.get("on_label_layer", True)
                       and self.label_head == "gaussian")   # CE head: logits are
            # observed, not a latent, so a label-layer prior makes no sense
            hlat = (Lateral(label_dim, min(lat["rank"], label_dim), lat["init_scale"],
                            lat["eps_ridge"], device) if use_lat else None)
            self.heads.append(Head(
                W=w_init(self.dims[self.L - 1], label_dim),
                b=torch.zeros(label_dim, device=device),
                log_pi=torch.full((label_dim,), float(p["log_pi_init"]), device=device),
                lateral=hlat))
        self.label_dim = label_dim

    # ------------------------------------------------------------------
    def layer_params(self, head_id):
        h = self.heads[head_id]
        Ws = self.W + [h.W]
        bs = self.b + [h.b]
        log_pis = self.log_pi + [h.log_pi]
        lats = self.laterals + [h.lateral]
        return Ws, bs, log_pis, lats

    def pis(self, head_id):
        _, _, log_pis, _ = self.layer_params(head_id)
        lo, hi = self.cfg["pc"]["log_pi_clamp"]
        return [torch.exp(lp.clamp(lo, hi)) for lp in log_pis]

    # NEW: categorical Fisher diagonal for the CE head, used as the label
    # layer's curvature block in the Schur-Fisher influence recursion.
    @torch.no_grad()
    def effective_label_pi(self, state, head_id):
        logits = state["mu"][-1]
        pr = torch.softmax(logits / self.ce_temp, dim=1).mean(dim=0)
        return (pr * (1 - pr) / self.ce_temp).clamp_min(1e-4)

    # ------------------------------------------------------------------
    def forward_init(self, x, head_id):
        Ws, bs, _, _ = self.layer_params(head_id)
        z = [x]
        for k in range(self.L):
            z.append(self.phi(z[k]) @ Ws[k].T + bs[k] if k > 0
                     else z[k] @ Ws[k].T + bs[k])
        return z

    def _pre(self, z, k):
        return z[0] if k == 0 else self.phi(z[k])

    # ------------------------------------------------------------------
    def _label_eps(self, z, mu, pis, y_onehot, label_mask):
        """Label-layer error under the configured head (see sign note above)."""
        if self.label_head == "ce":
            if y_onehot is None:
                e = torch.zeros_like(mu[-1])
            else:
                logits = mu[-1] / self.ce_temp
                if label_mask is not None:
                    # Restrict the softmax to ACTIVE classes. Without this,
                    # masked CE still inflates active logits until they beat
                    # every old class at argmax (softmax competition /
                    # task-recency bias) and old-task accuracy pins to 0.
                    logits = logits.masked_fill(label_mask == 0, -1e9)
                pr = torch.softmax(logits, dim=1)
                e = (y_onehot - pr) / self.ce_temp
        else:
            e = pis[-1] * (z[self.L] - mu[-1])
        if label_mask is not None:
            e = e * label_mask
        return e

    def infer(self, x, head_id, y_onehot=None, k_infer=None,
              record_energy=False, label_mask=None):
        """PC relaxation. Clamps z^0 = x always.

        gaussian head: clamps z^L = onehot when y given, relaxes it otherwise.
        ce head:       z^L is not a latent; it tracks the logits mu^L, and the
                       label error is the tempered softmax-CE logit gradient.
        label_mask:    (1, label_dim) 0/1 mask; zeroes label error for classes
                       outside the current task (class-IL protection of old
                       and future output weights).
        """
        p = self.cfg["pc"]
        K = k_infer or p["k_infer"]
        lr, mom, l1 = p["infer_lr"], p["polyak"], p["latent_l1"]
        Ws, bs, _, lats = self.layer_params(head_id)
        pis = self.pis(head_id)
        ce = (self.label_head == "ce")

        z = self.forward_init(x, head_id)
        clamp_label = (y_onehot is not None) and not ce
        if clamp_label:
            z[self.L] = y_onehot.clone()

        prev = [zi.clone() for zi in z]
        energies = []

        for _ in range(K):
            mu = [self._pre(z, k) @ Ws[k].T + bs[k] for k in range(self.L)]
            if ce:
                z[self.L] = mu[-1]                      # logits, not a latent
            eps = [pis[k] * (z[k + 1] - mu[k]) for k in range(self.L)]
            eps[-1] = self._label_eps(z, mu, pis, y_onehot, label_mask)

            if record_energy:
                e = sum(0.5 * ((z[k + 1] - mu[k]) ** 2 * pis[k]).sum()
                        for k in range(self.L))
                energies.append(float(e) / x.shape[0])

            new_z = [z[0]]
            # hidden layers always relax; gaussian free label relaxes too
            free_top = self.L if (clamp_label or ce) else self.L + 1
            for k in range(1, free_top):
                grad = eps[k - 1]
                if lats[k - 1] is not None:
                    grad = grad + lats[k - 1].matvec(z[k])
                if k < self.L:
                    # top-down credit: dF/dz^k -= J_k^T eps^{k+1}
                    grad = grad - self.dphi(z[k]) * (eps[k] @ Ws[k])
                zk = z[k] - lr * grad + mom * (z[k] - prev[k])
                if l1 > 0:
                    zk = zk - lr * l1 * torch.sign(zk)
                new_z.append(zk)
            if clamp_label or ce:
                new_z.append(z[self.L])
            prev = z
            z = new_z

        mu = [self._pre(z, k) @ Ws[k].T + bs[k] for k in range(self.L)]
        if ce:
            z[self.L] = mu[-1]
        eps = [pis[k] * (z[k + 1] - mu[k]) for k in range(self.L)]
        eps[-1] = self._label_eps(z, mu, pis, y_onehot, label_mask)
        out = {"z": z, "eps": eps, "mu": mu}
        if record_energy:
            out["energy"] = energies
        return out

    # ------------------------------------------------------------------
    def hebbian_deltas(self, state, head_id):
        z, eps = state["z"], state["eps"]
        B = z[0].shape[0]
        deltas, pres = [], []
        for k in range(self.L):
            pre = self._pre(z, k)
            dW = eps[k].T @ pre / B
            db = eps[k].mean(dim=0)
            deltas.append((dW, db))
            pres.append(pre)
        return deltas, pres

    def apply_deltas(self, deltas, head_id, lr, l2, l1, gates=None):
        Ws, bs, _, _ = self.layer_params(head_id)
        for k in range(self.L):
            dW, db = deltas[k]
            if gates is not None and gates[k] is not None:
                dW = gates[k] * dW
            dW = dW - l2 * Ws[k] - l1 * torch.sign(Ws[k])
            Ws[k].add_(lr * dW)
            bs[k].add_(lr * db)

    def precision_step(self, state, head_id, lr):
        z, mu = state["z"], state["mu"]
        _, _, log_pis, _ = self.layer_params(head_id)
        lo, hi = self.cfg["pc"]["log_pi_clamp"]
        top = self.L - 1 if self.label_head == "ce" else self.L  # CE: no
        # Gaussian residual at the label layer, so skip its precision
        for k in range(top):
            pi = torch.exp(log_pis[k].clamp(lo, hi))
            resid2 = ((z[k + 1] - mu[k]) ** 2).mean(dim=0)
            grad = 0.5 * pi * resid2 - 0.5
            log_pis[k].sub_(lr * grad).clamp_(lo, hi)

    def lateral_step(self, state, head_id, lr):
        z = state["z"]
        _, _, _, lats = self.layer_params(head_id)
        for k in range(1, self.L + 1):
            if lats[k - 1] is not None:
                lats[k - 1].learn_step(z[k].detach(), lr)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x, head_id):
        state = self.infer(x, head_id, y_onehot=None)
        return state["z"][self.L].argmax(dim=1)

    # ---- snapshots -------------------------------------------------------
    def snapshot(self):
        return {
            "W": [w.clone() for w in self.W],
            "b": [b.clone() for b in self.b],
            "log_pi": [p.clone() for p in self.log_pi],
            "lat": [l.B.clone() if l else None for l in self.laterals],
            "heads": [{"W": h.W.clone(), "b": h.b.clone(), "log_pi": h.log_pi.clone(),
                       "lat": h.lateral.B.clone() if h.lateral else None}
                      for h in self.heads],
        }

    def load_snapshot(self, s):
        for w, w0 in zip(self.W, s["W"]):
            w.copy_(w0)
        for b, b0 in zip(self.b, s["b"]):
            b.copy_(b0)
        for p, p0 in zip(self.log_pi, s["log_pi"]):
            p.copy_(p0)
        for l, l0 in zip(self.laterals, s["lat"]):
            if l is not None:
                l.B.copy_(l0)
        for h, h0 in zip(self.heads, s["heads"]):
            h.W.copy_(h0["W"]); h.b.copy_(h0["b"]); h.log_pi.copy_(h0["log_pi"])
            if h.lateral is not None:
                h.lateral.B.copy_(h0["lat"])

    def flat_params(self):
        parts = [w.flatten() for w in self.W] + [b.flatten() for b in self.b]
        for h in self.heads:
            parts += [h.W.flatten(), h.b.flatten()]
        return torch.cat(parts)
