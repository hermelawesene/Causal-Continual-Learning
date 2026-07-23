"""
trainers/pc_trainer.py
Trainer for PCNetwork (discriminative PC, no causal coding).

After pc_forward() and optimizer.step() (which updates l2 + head),
call hebbian_l1_update() to apply the local PC weight rule to l1.

FIX: original trainer only called optimizer.step(), which — because l1
weights are not in the logits computation graph — gave l1 zero gradient.
The Hebbian update fills this gap using stored equilibrium errors.
"""

import torch
from configs import CFG
from models import PCNetwork
from trainers.base_trainer import BaseTrainer


class PCTrainer(BaseTrainer):
    method_name = "PC Network"

    def __init__(self, train_loaders, test_loaders, task_names, device,
                 n_epochs=CFG.N_EPOCHS, lr=CFG.LR, n_modules=CFG.N_MODULES):
        model = PCNetwork(n_tasks=len(train_loaders), n_modules=n_modules)
        super().__init__(model, train_loaders, test_loaders, task_names,
                         device, n_epochs, lr)

    def train_one_task(self, task_id: int):
        self.model.train()
        for _ in range(self.n_epochs):
            for x, y in self.train_loaders[task_id]:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                # PC forward: refines z, returns logits differentiable through l2+head
                logits, _, _ = self.model.pc_forward(x, task_id, y)

                loss = self.criterion(logits, y)
                loss.backward()          # updates l2 weights + head via backprop
                self.optimizer.step()

                # Hebbian update for l1 (local PC weight rule, no backprop needed)
                self.model.hebbian_l1_update()