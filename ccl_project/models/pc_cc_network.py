"""
models/pc_cc_network.py
Discriminative PC + Causal Coding network.

Extends PCNetwork with:

1. CAUSAL GATES
   For module m and task t:
       influence(m) = mean_batch ||∂CE(logits, y)/∂h_m||
   This is the task-loss Jacobian norm — measures causal contribution to
   CORRECT classification, not just output sensitivity.

   Normalised and sharpened, then clamped to [0, 1]:
       gate(m) = clamp(normalised_score(m), 0, 1)
   Gates in [0,1] ONLY suppress updates (never amplify).
   A gate > 1 would amplify the dominant module's gradient, destabilising
   learning — clamping prevents this.

   Applied to backbone gradients after loss.backward():
       grad[theta_m] *= gate[m]

   FIX vs previous: previous _norm() could produce gates > N (e.g. gate=4
   for 4 modules when one module dominates). This amplified gradients for
   the dominant module instead of suppressing others. Clamping to [0,1]
   fixes this: inactive modules are frozen, active modules update at full rate.

2. CLARITY PENALTY
   L1 penalty on off-diagonal weight blocks in L2 only.
   L2 module m_out has weight (module_dim, hidden_dim).
   Penalise blocks W[:, m_in*d:(m_in+1)*d] for m_in != m_out.

   FIX vs previous: previous version also penalised L1 off-diagonal input
   slices (pixels 0-195 vs 196-391 etc.). This is wrong for MNIST: digits
   span the full image, not quadrant-sized pixel chunks. Forcing each L1
   module to only read from a pixel quadrant actively degrades representation
   quality with no theoretical benefit — the paper's modularity argument is
   about FEATURE modularity, not input-pixel slicing. L1 clarity removed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import CFG
from models.pc_network import PCNetwork


class PCCCNetwork(PCNetwork):

    def __init__(
        self,
        input_dim  = CFG.INPUT_DIM,
        hidden_dim = CFG.HIDDEN_DIM,
        n_tasks    = CFG.N_TASKS,
        n_modules  = CFG.N_MODULES,
        pc_iters   = 10,
        lr_z       = 0.1,
        gate_power = CFG.GATE_POWER,
    ):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim,
                         n_tasks=n_tasks, n_modules=n_modules,
                         pc_iters=pc_iters, lr_z=lr_z)
        self.gate_power = gate_power
        self._gates_l1  = torch.ones(n_modules)
        self._gates_l2  = torch.ones(n_modules)

    # ------------------------------------------------------------------
    def estimate_gates(self, x_batch, task_id, y_batch=None, batch_size=CFG.GATE_BATCH):
        """
        Task-loss Jacobian gate estimation.

        For each module m:
            score(m) = mean_batch ||∂CE(logits,y)/∂h_m||

        Normalised to [0,1] range:
            gate(m) = clamp(normalised_sharpened_score(m), 0, 1)

        FIX 1: use CE(logits,y) not logits.sum() — label-aware influence.
        FIX 2: clamp gates to [0,1] — suppress low-influence modules only,
                never amplify dominant ones.
        """
        was_training = self.training
        self.eval()

        x  = x_batch[:batch_size].detach()
        xf = x.view(x.size(0), -1)
        use_loss = y_batch is not None
        y  = y_batch[:batch_size].detach() if use_loss else None
        criterion = nn.CrossEntropyLoss()

        inf_l1, inf_l2 = [], []

        # ── Layer 1 ───────────────────────────────────────────────────────
        for mi in range(self.n_modules):
            self.zero_grad()
            hm_leaf, parts = None, []
            for j, mod in enumerate(self.l1):
                act = F.relu(mod(xf))
                if j == mi:
                    hm_leaf = act.detach().requires_grad_(True)
                    parts.append(hm_leaf)
                else:
                    parts.append(act.detach())
            h1     = torch.cat(parts, dim=1)
            h2     = torch.cat([F.relu(m(h1)) for m in self.l2], dim=1)
            logits = self.heads[task_id](h2)
            signal = criterion(logits, y) if use_loss else logits.sum()
            signal.backward()
            score = (hm_leaf.grad.norm(dim=1).mean().item()
                     if (hm_leaf is not None and hm_leaf.grad is not None) else 1e-8)
            inf_l1.append(score)

        # ── Layer 2 ───────────────────────────────────────────────────────
        with torch.no_grad():
            h1_det = torch.cat([F.relu(m(xf)) for m in self.l1], dim=1)

        for mi in range(self.n_modules):
            self.zero_grad()
            hm_leaf, parts = None, []
            for j, mod in enumerate(self.l2):
                act = F.relu(mod(h1_det))
                if j == mi:
                    hm_leaf = act.detach().requires_grad_(True)
                    parts.append(hm_leaf)
                else:
                    parts.append(act.detach())
            h2     = torch.cat(parts, dim=1)
            logits = self.heads[task_id](h2)
            signal = criterion(logits, y) if use_loss else logits.sum()
            signal.backward()
            score = (hm_leaf.grad.norm(dim=1).mean().item()
                     if (hm_leaf is not None and hm_leaf.grad is not None) else 1e-8)
            inf_l2.append(score)

        self.zero_grad()
        if was_training:
            self.train()

        def _norm_and_clamp(scores):
            import torch as _t
            t = _t.tensor(scores, dtype=_t.float32).clamp(min=1e-8)
            k = 1
            top_idx = torch.topk(t, k).indices
            mask = torch.zeros_like(t)
            mask[top_idx] = 1.0
            t = mask
            #
            # Normalise so mean = 1
            # t = t / t.sum() * len(scores)
            # # Sharpen: high-influence modules get > 1, low-influence get < 1
            # t = t ** self.gate_power
            # # Re-normalise so mean = 1 again (mean gate = 1 -> average update unchanged)
            # t = t / t.sum() * len(scores)
            # # CLAMP to [0, 1]: gates only suppress, never amplify
            # # This means the average gate < 1 overall, which is fine —
            # # we want the high-influence modules to update at full rate (1.0)
            # # and low-influence modules to be suppressed (< 1.0).
            # t = t.clamp(max=1.0)
            return t.cpu()

        self._gates_l1 = _norm_and_clamp(inf_l1)
        self._gates_l2 = _norm_and_clamp(inf_l2)
        return self._gates_l1, self._gates_l2

    # ------------------------------------------------------------------
    def apply_gates(self, gates_l1=None, gates_l2=None):
        """Scale backbone gradients by gate values (in [0,1]) after loss.backward()."""
        g1 = gates_l1 if gates_l1 is not None else self._gates_l1
        g2 = gates_l2 if gates_l2 is not None else self._gates_l2
        for mi, mod in enumerate(self.l1):
            s = g1[mi].item()
            for p in mod.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(s)
        for mi, mod in enumerate(self.l2):
            s = g2[mi].item()
            for p in mod.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(s)

    # ------------------------------------------------------------------
    def clarity_penalty(self):
        """
        L1 penalty on off-diagonal weight blocks in L2 ONLY.

        FIX: removed L1 clarity. Penalising L1 input slices forces each module
        to read from only a pixel quadrant of the 28x28 image. MNIST digits
        span the whole image — this constraint actively hurts representation
        quality with no theoretical backing from the CCL paper.
        The paper's support-set argument (S_t = set of causally active modules)
        applies to FEATURE modularity at the hidden layer, not to raw input pixels.
        """
        device  = next(self.parameters()).device
        penalty = torch.zeros(1, device=device)
        for mi_out, mod in enumerate(self.l2):
            W = mod.weight   # (module_dim, hidden_dim)
            for mi_in in range(self.n_modules):
                if mi_in == mi_out:
                    continue
                s = mi_in * self.module_dim
                e = (mi_in + 1) * self.module_dim
                penalty = penalty + W[:, s:e].abs().mean()
        return penalty

    # ------------------------------------------------------------------
    def get_current_gates(self):
        return self._gates_l1.clone(), self._gates_l2.clone()