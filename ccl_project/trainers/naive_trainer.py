"""
trainers/naive_trainer.py
Trainer for NaiveMLP — plain Adam + CrossEntropyLoss.
No gating, no special treatment. Full backbone updates every task.
"""

import torch
from configs import CFG
from models import NaiveMLP
from trainers.base_trainer import BaseTrainer


class NaiveTrainer(BaseTrainer):
    method_name = "Naive MLP"

    def __init__(self, train_loaders, test_loaders, task_names, device,
                 n_epochs=CFG.N_EPOCHS, lr=CFG.LR):
        model = NaiveMLP(n_tasks=len(train_loaders))
        super().__init__(model, train_loaders, test_loaders, task_names,
                         device, n_epochs, lr)

    def train_one_task(self, task_id: int):
        self.model.train()
        for _ in range(self.n_epochs):
            for x, y in self.train_loaders[task_id]:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(x, task_id), y)
                #loss = self.criterion(self.model(x), y)
                loss.backward()
                self.optimizer.step()
