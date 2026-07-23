"""Do-influence estimation for the discriminative PC network.

(See v1 docstring for the full generative->discriminative mirror derivation;
unchanged.)

CHANGE (v2): CE label head support. When net.label_head == "ce" the label
layer is an observed categorical, not a Gaussian latent, so its curvature
block in the Schur-Fisher recursion is the categorical Fisher. We use the
diagonal Gauss-Newton approximation

    Pi_L^eff = diag( p (1 - p) / T ),      p = softmax(mu_L / T) batch-avg,

which is exactly the natural Gauss-Newton object the influence note is built
on, and drop the label-layer lateral prior (meaningless for observed logits;
the network builds the CE head without one).

1. COMPOSED INFLUENCE (Schur-Fisher, mirrored to the label):
        G_k    = Pi_k + Lam_k + J_k^T Pi_{k+1} J_k     (hidden layers)
        G_L    = Pi_L^eff (+ Lam_L for gaussian head)
        Gbar_L = G_L;   Gbar_k = G_k - J_k^T Pi_{k+1} Gbar_{k+1}^{-1} Pi_{k+1} J_k
        A_k    = Gbar_{k+1}^{-1} Pi_{k+1} J_k
        C^{l->L} = A_{L-1} ... A_l          (+ ridge on every Gbar)

2. PER-WEIGHT PROXY M_l (pseudocode 3.2):
        M_l = row_scale(child derivs) * |W_l| * col_scale(parent stats)
"""
from __future__ import annotations

from typing import List

import torch


# ----------------------------------------------------------------------
@torch.no_grad()
def composed_influence(net, state, head_id: int, ridge: float = 1e-4):
    """Local maps A_k, composed C^{l->L} (l = 1..L-1), per-unit label scores."""
    Ws, _, _, lats = net.layer_params(head_id)
    pis = net.pis(head_id)
    ce = getattr(net, "label_head", "gaussian") == "ce"
    if ce:
        pis = pis[:-1] + [net.effective_label_pi(state, head_id)]   # NEW
    z = state["z"]
    L = net.L
    dev = z[0].device

    dphis = []
    for k in range(L):
        d = torch.ones(net.dims[0], device=dev) if k == 0 else net.dphi(z[k]).mean(dim=0)
        dphis.append(d)

    def Lam_full(k):
        lat = lats[k - 1]
        if lat is None:
            return torch.zeros(net.dims[k], net.dims[k], device=dev)
        return lat.full()

    def J(k):
        return Ws[k] * dphis[k].unsqueeze(0)

    eye = lambda n: torch.eye(n, device=dev)

    G = {}
    for k in range(1, L + 1):
        Gk = torch.diag(pis[k - 1])
        if not (ce and k == L):           # CE head: no lateral prior on logits
            Gk = Gk + Lam_full(k)
        if k < L:
            Jk = J(k)
            Gk = Gk + Jk.T @ (pis[k].unsqueeze(1) * Jk)
        G[k] = Gk

    Gbar = {L: G[L] + ridge * eye(net.dims[L])}
    for k in range(L - 1, 0, -1):
        Jk = J(k)
        PJ = pis[k].unsqueeze(1) * Jk
        Gbar[k] = G[k] - Jk.T @ torch.linalg.solve(Gbar[k + 1], PJ)
        Gbar[k] = Gbar[k] + ridge * eye(net.dims[k])

    A = {}
    for k in range(0, L):
        PJ = pis[k].unsqueeze(1) * J(k)
        A[k] = torch.linalg.solve(Gbar[k + 1], PJ)

    C = {}
    for l in range(1, L):
        M = A[l]
        for k in range(l + 1, L):
            M = A[k] @ M
        C[l] = M

    unit_scores = {l: C[l].abs().sum(dim=0) for l in C}
    return {"A": A, "C": C, "unit_scores": unit_scores}


# ----------------------------------------------------------------------
@torch.no_grad()
def weight_proxy_M(net, state, pres: List[torch.Tensor], head_id: int, eps: float = 1e-6):
    """Naturalized per-weight influence proxy M_l (pseudocode step 3.2)."""
    Ws, bs, _, _ = net.layer_params(head_id)
    Ms = []
    for k in range(net.L):
        a_child = state["mu"][k]
        der = net.dphi(a_child) if k + 1 <= net.L else torch.ones_like(a_child)
        s1 = der.abs().mean(dim=0)
        s2 = (der ** 2).mean(dim=0) + eps
        row_scale = (s1 / s2).unsqueeze(1)
        col_scale = (1.0 / torch.sqrt((pres[k] ** 2).mean(dim=0) + eps)).unsqueeze(0)
        Ms.append(row_scale * Ws[k].abs() * col_scale)
    return Ms
