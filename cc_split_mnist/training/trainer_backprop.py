"""Backprop baseline: same architecture as the PC nets, trained with Adam+CE.

CHANGE (v2): loss.item() instead of float(loss) (autograd warning fix).
Note: BP intentionally gets NO label masking or protection — it is the
unprotected fine-tuning baseline every CL paper compares against.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BPNet(nn.Module):
    def __init__(self, cfg, label_dim, n_heads):
        super().__init__()
        m = cfg["model"]
        act = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid}[m["activation"]]
        dims = [m["input_dim"]] + list(m["hidden_dims"])
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), act()]
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(dims[-1], label_dim) for _ in range(n_heads)])

    def forward(self, x, head_id):
        return self.heads[head_id](self.trunk(x))


class BackpropTrainer:
    name = "bp"

    def __init__(self, cfg, label_dim, n_heads, device):
        self.cfg = cfg
        self.device = device
        self.net = BPNet(cfg, label_dim, n_heads).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg["bp"]["lr"])
        self.crit = nn.CrossEntropyLoss()
        self.epochs = cfg["bp"]["epochs_per_task"]

    def train_task(self, loader, head_id, task_id, logger=None):
        self.net.train()
        for ep in range(self.epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                self.opt.zero_grad()
                loss = self.crit(self.net(x, head_id), y)
                loss.backward()
                self.opt.step()
            if logger:
                logger(task_id, ep, {"loss": loss.item()})

    def train_batches(self, batches, head_id):
        self.net.train()
        for x, y in batches:
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            self.crit(self.net(x, head_id), y).backward()
            self.opt.step()

    @torch.no_grad()
    def evaluate(self, loader, head_id):
        self.net.eval()
        correct = total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.net(x, head_id).argmax(dim=1)
            correct += int((pred == y).sum())
            total += len(y)
        return correct / max(1, total)

    def flat_params(self):
        return torch.cat([p.detach().flatten() for p in self.net.parameters()])

    def snapshot(self):
        return {k: v.clone() for k, v in self.net.state_dict().items()}

    def load_snapshot(self, s):
        self.net.load_state_dict(s)

    def module_influence(self, probe, head_id):
        return None
