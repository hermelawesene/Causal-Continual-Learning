"""
trainers/cc_trainer.py
Trainer for PCCCNetwork (discriminative PC + Causal Coding).

Per training step:
  1. Estimate causal gates from task-loss Jacobian norms (every GATE_FREQ steps)
     Pass y so estimate_gates() uses cross-entropy loss, not logits.sum()
  2. pc_forward() — discriminative PC inference, returns logits via l2+head
  3. loss = CrossEntropy(logits, y) + clarity_weight * clarity_penalty()
  4. loss.backward()
  5. apply_gates() — scale backbone gradients by gate values in [0,1]
  6. optimizer.step()  — updates l2 + head
  7. hebbian_l1_update(gate_scales) — updates l1 with gated Hebbian rule

Why this bounds forgetting (CCL theorem):
  - apply_gates() enforces epsilon_g ~ 0  (out-of-support gradient near zero)
  - clarity_penalty() enforces epsilon_h ~ 0  (cross-module l2 Hessian near zero)
  - Together: epsilon_comm ~ 0  => Theorem 1 bounds sequential vs joint training gap
"""

import torch
from configs import CFG
from models import PCCCNetwork
from trainers.base_trainer import BaseTrainer


class CCTrainer(BaseTrainer):
    method_name = "PC + Causal Coding"

    def __init__(self, train_loaders, test_loaders, task_names, device,
                 n_epochs=CFG.N_EPOCHS, lr=CFG.LR,
                 n_modules=CFG.N_MODULES, clarity_w=CFG.CLARITY_W,
                 gate_freq=CFG.GATE_FREQ):
        model = PCCCNetwork(n_tasks=len(train_loaders), n_modules=n_modules)
        super().__init__(model, train_loaders, test_loaders, task_names,
                         device, n_epochs, lr)
        self.clarity_w = clarity_w
        self.gate_freq = gate_freq
        self.gate_history_l1: list = []
        self.gate_history_l2: list = []

    def train_one_task(self, task_id: int):
        self.model.train()
        step = 0

        for _ in range(self.n_epochs):
            for x, y in self.train_loaders[task_id]:
                x, y = x.to(self.device), y.to(self.device)

                # 1. Re-estimate gates (task-loss Jacobian, label-aware)
                if step % self.gate_freq == 0:
                    self.model.estimate_gates(x, task_id, y_batch=y)
                    self.model.train()

                self.optimizer.zero_grad()

                # 2. Discriminative PC forward (correct inference dynamics)
                logits, _, _ = self.model.pc_forward(x, task_id, y)

                # 3. Loss + clarity penalty (l2 off-diagonal blocks only)
                loss = (self.criterion(logits, y)
                        + self.clarity_w * self.model.clarity_penalty())

                # 4. Backward
                loss.backward()

                # 5. Apply gates to backbone gradients (gates in [0,1])
                self.model.apply_gates()

                # 6. Optimizer step — updates l2 + head
                self.optimizer.step()

                # 7. Gated Hebbian update for l1
                gate_scales_l1 = self.model._gates_l1.tolist()
                self.model.hebbian_l1_update(gate_scales=gate_scales_l1)

                step += 1

        g1, g2 = self.model.get_current_gates()
        self.gate_history_l1.append(g1.numpy())
        self.gate_history_l2.append(g2.numpy())
        self.results.gate_history = list(zip(self.gate_history_l1, self.gate_history_l2))