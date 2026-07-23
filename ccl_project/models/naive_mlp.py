"""
models/naive_mlp.py
Standard two-hidden-layer MLP with per-task output heads.

This is the control condition: the backbone receives full unrestricted
gradients on every task, so we expect heavy catastrophic forgetting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import CFG


class NaiveMLP(nn.Module):
    """
    Input(784) -> Linear(400) -> ReLU -> Linear(400) -> ReLU -> heads[task_id](2)

    Per-task heads prevent the output layer from being overwritten each task,
    so all forgetting originates in the shared backbone — exactly what the
    CCL theorem talks about.
    """

    def __init__(self, input_dim=CFG.INPUT_DIM, hidden_dim=CFG.HIDDEN_DIM, n_tasks=CFG.N_TASKS):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(n_tasks)])
        #self.heads = nn.ModuleList([nn.Linear(hidden_dim, 10) for _ in range(n_tasks)])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x, task_id):
        h = self.backbone(x.view(x.size(0), -1))
        return self.heads[task_id](h)
        #return self.heads(h)
