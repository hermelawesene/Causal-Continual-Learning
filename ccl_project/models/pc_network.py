"""
models/pc_network.py
Discriminative Predictive Coding (PC) network with a modular backbone.

PC INFERENCE — CORRECT z DYNAMICS
===================================
Standard discriminative PC (Whittington & Bogacz 2017, eq. 8-9):

    e1 = z1 - f1(x)               # layer-1 prediction error
    e2 = z2 - f2(z1)              # layer-2 prediction error

    dz1/dt = -e1  +  W2^T * e2 * sigma'(z1)   <-- top-down error term
    dz2/dt = -e2  +  d(task_loss)/dz2          <-- output error term

The top-down term  W2^T * e2  flows from e2 back into z1.
This requires pred2 = f2(z1) to stay in the computation graph w.r.t. z1
so that autograd.grad(energy, z1) picks up both  de1/dz1  AND  de2/dz1.

BUG IN PREVIOUS VERSION:
    pred2 = self._l2_forward(z1).detach()   # <-- WRONG: kills top-down signal
    Result: dz1/dt = -e1 only.  z1 never receives top-down correction.
    z1 converges to the feedforward value f1(x), not the PC equilibrium.

FIX:
    pred2 = self._l2_forward(z1)            # NO detach: z1 stays in graph
    autograd.grad(energy, [z1, z2]) now correctly computes:
        grad_z1 = e1  -  W2^T * e2 * sigma'(z1)
        grad_z2 = e2  +  d(CE)/dz2
    W never gets updated here (we only pass [z1,z2] to grad(), not W).

WEIGHT UPDATE — CORRECT GRADIENT PATH
=======================================
After inference converges to z1_eq, z2_eq (both detached):

    h1_grad = l1(x)               # differentiable through l1 weights
    h2_grad = l2(z1_eq)           # differentiable through l2 weights
    logits  = head(h2_grad)       # differentiable through head + l2
    loss    = CE(logits, y)
    loss.backward()               # updates head, l2 weights

    l1 weight update uses the prediction error at equilibrium:
    ΔW1 ∝  e1_eq  *  x^T   (local Hebbian rule, Whittington & Bogacz eq. 5)
    We implement this as a separate Hebbian step, not backprop.

BUG IN PREVIOUS VERSION:
    h1_grad = l1(x)  was computed but NEVER entered the logits graph.
    logits = head(l2(z1_eq))  — z1_eq detached, so l2 got grad but l1 did NOT.
    h1_grad floated disconnected.  l1 weights got zero gradient every step.

FIX:
    Use a Hebbian update for l1: ΔW1 ∝ e1_eq * x^T
    This is the theoretically correct PC weight rule for the bottom layer,
    and avoids needing h1_grad in the logits graph.
    Alternatively (implemented here for simplicity and Adam compatibility):
    connect h1_grad into the logits computation by re-running l2 from h1_grad:
        h2_for_l1 = l2(h1_grad.detach())   -- gives l1 a grad path via h1_grad
        total_logits = head(h2_grad + 0*h2_for_l1)  -- attaches l1 to graph
    But this mixes two forward passes.  Cleaner: pass h1_grad through head too:
        logits = head(l2(h1_grad))  -- BUT this does not use z1_eq for l2.

    CLEANEST CORRECT APPROACH (implemented below):
    Run TWO weight-update passes, one per layer:
        Pass A: logits_A = head(l2(z1_eq))  ->  loss_A.backward()  -> updates l2, head
        Pass B: logits_B = head(l2(h1_grad).detach())  WRONG, still disconnects

    ACTUALLY CORRECT AND SIMPLE:
    The PC weight rule (W&B eq. 5) for layer 1 is:
        ΔW1 ∝  e1_eq  *  x^T
    where e1_eq = z1_eq - W1*x  (evaluated at equilibrium).
    We just compute this directly and apply it.  For l2 we use backprop.
    This matches the theory exactly and avoids the graph-connection puzzle.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import CFG


class PCNetwork(nn.Module):
    """
    Modular, discriminative PC network.
    Same forward(x, task_id) interface as NaiveMLP.
    """

    def __init__(
        self,
        input_dim  = CFG.INPUT_DIM,
        hidden_dim = CFG.HIDDEN_DIM,
        n_tasks    = CFG.N_TASKS,
        n_modules  = CFG.N_MODULES,
        pc_iters   = 10,
        lr_z       = 0.1,
        lr_w_pc    = 1e-3,   # learning rate for Hebbian l1 update
    ):
        super().__init__()
        assert hidden_dim % n_modules == 0
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.n_modules  = n_modules
        self.module_dim = hidden_dim // n_modules
        self.n_tasks    = n_tasks
        self.pc_iters   = pc_iters
        self.lr_z       = lr_z
        self.lr_w_pc    = lr_w_pc

        self.l1    = nn.ModuleList([nn.Linear(input_dim,  self.module_dim) for _ in range(n_modules)])
        self.l2    = nn.ModuleList([nn.Linear(hidden_dim, self.module_dim) for _ in range(n_modules)])
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(n_tasks)])
        self._init_weights()

        # Store last equilibrium errors for Hebbian update
        self._last_e1_eq: list = []   # per-module, shape (B, module_dim)
        self._last_x_eq:  torch.Tensor = None

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def _l1_forward(self, x):
        xf = x.view(x.size(0), -1)
        return torch.cat([F.relu(m(xf)) for m in self.l1], dim=1)

    def _l2_forward(self, h1):
        return torch.cat([F.relu(m(h1)) for m in self.l2], dim=1)

    def forward(self, x, task_id):
        """Plain feedforward — used at eval time."""
        h1 = self._l1_forward(x)
        h2 = self._l2_forward(h1)
        return self.heads[task_id](h2)

    def pc_forward(self, x, task_id, y=None):
        """
        Discriminative PC forward with correct inference dynamics and weight updates.

        INFERENCE (W fixed, z moves):
          pred1 = l1(x).detach()      -- W1 gets no grad (correct: W fixed in inference)
          pred2 = l2(z1)              -- NO detach: z1 stays in graph for top-down signal
          e1 = z1 - pred1
          e2 = z2 - pred2
          energy = 0.5||e1||^2 + 0.5||e2||^2 + CE(head(z2), y)
          autograd.grad(energy, [z1, z2]):
              grad_z1 = e1 - W2^T*e2*sigma'(z1)   <-- correct top-down term
              grad_z2 = e2 + grad_CE_wrt_z2

        WEIGHT UPDATE:
          l2 + head: backprop through l2(z1_eq) -> head -> CE loss
          l1:        Hebbian rule: ΔW1_m ∝ e1_eq_m * xf^T  (W&B eq. 5)
                     Applied in hebbian_l1_update() called by trainer.

        Returns:
          logits  : (B, 2)  from l2(z1_eq) -> head  [differentiable through l2, head]
          z1_eq   : equilibrium z1 (detached)
          z2_eq   : equilibrium z2 (detached)
        """
        if y is None or not self.training:
            return self.forward(x, task_id), None, None

        criterion = nn.CrossEntropyLoss()
        xf = x.view(x.size(0), -1)

        # ── Feedforward initialisation ─────────────────────────────────────
        with torch.no_grad():
            z1 = self._l1_forward(x)
            z2 = self._l2_forward(z1)

        z1 = z1.detach().requires_grad_(True)
        z2 = z2.detach().requires_grad_(True)

        # ── Inference loop: W fixed, z moves ──────────────────────────────
        for _ in range(self.pc_iters):
            # pred1: detach from W1 so W1 gets no grad during inference
            pred1 = self._l1_forward(x).detach()

            # pred2: NO detach — z1 must stay in graph to get top-down error
            # W2 will appear in grad_z1 as  W2^T * e2 * sigma'(z1)
            # W2 does NOT get updated here because we only call grad([z1,z2])
            pred2 = self._l2_forward(z1)

            e1 = z1 - pred1
            e2 = z2 - pred2

            logits_inf = self.heads[task_id](z2)
            task_loss  = criterion(logits_inf, y)

            energy = (
                0.5 * (e1 ** 2).sum(1).mean()
                + 0.5 * (e2 ** 2).sum(1).mean()
                + task_loss
            )

            # grad w.r.t. z only — W is not in this list, so W never gets updated
            grads = torch.autograd.grad(energy, [z1, z2], create_graph=False)

            with torch.no_grad():
                z1 = (z1 - self.lr_z * grads[0]).detach().requires_grad_(True)
                z2 = (z2 - self.lr_z * grads[1]).detach().requires_grad_(True)

        z1_eq = z1.detach()
        z2_eq = z2.detach()

        # ── Store equilibrium errors for Hebbian l1 update ─────────────────
        with torch.no_grad():
            self._last_x_eq = xf.detach()
            self._last_e1_eq = []
            for mi, mod in enumerate(self.l1):
                s = mi * self.module_dim
                e = (mi + 1) * self.module_dim
                pred1_m = F.relu(mod(xf))
                e1_m    = z1_eq[:, s:e] - pred1_m
                self._last_e1_eq.append(e1_m.detach())

        # ── Weight update pass for l2 + head (backprop) ───────────────────
        # z1_eq is fixed (detached) input to l2 — gives l2 weights a grad path
        h2_grad = self._l2_forward(z1_eq)          # differentiable through l2
        logits  = self.heads[task_id](h2_grad)      # differentiable through head + l2

        return logits, z1_eq, z2_eq

    def hebbian_l1_update(self, gate_scales=None):
        """
        Apply Hebbian weight update to l1 using stored equilibrium errors.

        ΔW1_m = lr * e1_eq_m^T * xf / B     (Whittington & Bogacz 2017, eq. 5)

        gate_scales: list of N floats in [0,1] — scale update per module.
                     None means all scales = 1.
        Call this AFTER pc_forward() and optimizer.step().
        """
        if self._last_x_eq is None or len(self._last_e1_eq) == 0:
            return
        xf = self._last_x_eq    # (B, input_dim)
        B  = xf.size(0)
        with torch.no_grad():
            for mi, mod in enumerate(self.l1):
                e1_m = self._last_e1_eq[mi]   # (B, module_dim)
                scale = gate_scales[mi] if gate_scales is not None else 1.0
                if scale < 1e-8:
                    continue
                # ΔW = (e1_m^T @ xf) / B   shape: (module_dim, input_dim)
                dW = (e1_m.T @ xf) / B
                mod.weight.data.add_(self.lr_w_pc * scale * dW)

    def get_backbone_params(self):
        return list(self.l1.parameters()) + list(self.l2.parameters())