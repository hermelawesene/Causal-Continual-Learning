"""Clarity gradient (pseudocode doc, step 3.4).

Penalize edges overshadowed by multi-hop diffusion routes through the
influence graph, pushing weight mass toward direct causal paths. Tractable
specialization: truncated diffusion through the child layer's lateral graph.

    P_l = row-normalized M_l;  K = avg_{t=2..order} (Lmat^t-hop applied to P)
    os  = relu(K - P - delta);  D_l = os * P_l * sign(W_l)
    dW_l <- dW_l - lambda_diff * D_l   (shrinks overshadowed edges)
"""
from __future__ import annotations

from typing import List, Optional

import torch

from core.gates import normalize_inputwise


@torch.no_grad()
def clarity_grads(net, Ms: List[torch.Tensor], head_id: int,
                  order: int = 3, delta: float = 0.01):
    Ws, _, _, lats = net.layer_params(head_id)
    Ds: List[Optional[torch.Tensor]] = []
    for k in range(net.L):
        P = normalize_inputwise(Ms[k])
        lat = lats[k]
        if lat is not None:
            Lmat = lat.full().abs()
            Lmat = Lmat / (Lmat.sum(dim=1, keepdim=True) + 1e-8)
        else:
            Lmat = torch.eye(P.shape[0], device=P.device)
        K = torch.zeros_like(P)
        hop = P.clone()
        for _ in range(2, order + 1):
            hop = Lmat @ hop
            K = K + hop
        K = K / max(1, order - 1)
        os = torch.relu(K - P - delta)
        Ds.append(os * P * torch.sign(Ws[k]))
    return Ds
