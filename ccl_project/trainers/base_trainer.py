"""
trainers/base_trainer.py
Shared result tracking and base training loop.
All concrete trainers inherit from BaseTrainer.
"""

from __future__ import annotations
import time
import copy
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs import CFG


class Results:
    """
    acc_matrix[i][j] = accuracy on task j right after finishing training on task i.
    """
    def __init__(self, name: str, n_tasks: int, task_names: List[str]):
        self.name        = name
        self.n_tasks     = n_tasks
        self.task_names  = task_names
        self.acc_matrix  = np.zeros((n_tasks, n_tasks))
        self.train_time: List[float] = []
        self.gate_history: list = []   # filled by CCTrainer

    def avg_accuracy_after(self, task_idx: int) -> float:
        return float(np.mean(self.acc_matrix[task_idx, : task_idx + 1]))

    def backward_transfer(self) -> float:
        vals = [
            self.acc_matrix[i, j] - self.acc_matrix[j, j]
            for i in range(1, self.n_tasks)
            for j in range(i)
        ]
        return float(np.mean(vals)) if vals else 0.0

    def forgetting(self, task_idx: int = 0) -> float:
        return float(self.acc_matrix[-1, task_idx] - self.acc_matrix[task_idx, task_idx])

    def final_avg_accuracy(self) -> float:
        return float(np.mean(self.acc_matrix[-1, :]))

    def print_summary(self):
        w = 12
        sep = "=" * (w * (self.n_tasks + 1) + 4)
        print(f"\n{sep}")
        print(f"  {self.name}")
        print(sep)
        header = f"{'After':>{w}}" + "".join(f"{'T'+str(j+1):>{w}}" for j in range(self.n_tasks))
        print(header)
        print("-" * len(header))
        for i in range(self.n_tasks):
            row = f"{'Task '+str(i+1):>{w}}"
            for j in range(self.n_tasks):
                row += f"{self.acc_matrix[i, j]:>{w}.1f}%" if j <= i else f"{'—':>{w}}"
            print(row)
        print(f"\n  BWT               : {self.backward_transfer():+.2f}%")
        print(f"  Task-1 forgetting : {self.forgetting(0):+.2f}%")
        print(f"  Final avg accuracy: {self.final_avg_accuracy():.1f}%")
        print(sep)


class BaseTrainer:
    method_name: str = "base"

    def __init__(self, model, train_loaders, test_loaders, task_names, device,
                 n_epochs=CFG.N_EPOCHS, lr=CFG.LR):
        self.model         = model.to(device)
        self.train_loaders = train_loaders
        self.test_loaders  = test_loaders
        self.task_names    = task_names
        self.device        = device
        self.n_tasks       = len(train_loaders)
        self.n_epochs      = n_epochs
        self.criterion     = nn.CrossEntropyLoss()
        self.optimizer     = torch.optim.Adam(model.parameters(), lr=lr)
        self.results       = Results(self.method_name, self.n_tasks, task_names)

    def evaluate(self, loader, task_id, seen_tasks=None) -> float:
        """
        Evaluate on one task's test set.

        Task-Incremental (TIL): task_id selects the correct head; labels
            are always 0/1 within that head.
        Class-Incremental (CIL): all heads seen so far are concatenated into
            one big logit vector; labels are remapped to global class indices
            (task 0 → 0,1 | task 1 → 2,3 | …).
        """
        from configs import CFG
        self.model.eval()
        correct = total = 0

        cil = (CFG.INCREMENTAL_MODE == "class")
        if cil and seen_tasks is None:
            seen_tasks = list(range(task_id + 1))

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                if not cil:
                    # TIL: use the single correct head
                    logits = self.model(x, task_id)
                    preds  = logits.argmax(1)
                    correct += (preds == y).sum().item()
                else:
                    # CIL: concatenate all seen heads → global logit vector
                    all_logits = torch.cat(
                        [self.model(x, t) for t in seen_tasks], dim=1
                    )  # shape: (B, 2 * len(seen_tasks))
                    preds_global = all_logits.argmax(1)

                    # remap local labels (0/1) to global class indices
                    classes_in_task = 2  # always binary per task here
                    global_offset   = task_id * classes_in_task
                    y_global        = y + global_offset

                    correct += (preds_global == y_global).sum().item()

                total += y.size(0)

        return 100.0 * correct / total

    def train_one_task(self, task_id: int):
        raise NotImplementedError

    def run(self, verbose=True) -> Results:
        for t in range(self.n_tasks):
            if verbose:
                print(f"  [{self.method_name:22s}] Task {t+1}/{self.n_tasks}"
                      f"  ({self.task_names[t]}) ...", end=" ", flush=True)
            t0 = time.time()
            self.train_one_task(t)
            elapsed = time.time() - t0
            self.results.train_time.append(elapsed)
            seen = list(range(t + 1))
            for ev in range(self.n_tasks):
                self.results.acc_matrix[t, ev] = self.evaluate(
                    self.test_loaders[ev], ev, seen_tasks=seen
                )
            if verbose:
                seen_accs = [f"{self.results.acc_matrix[t, j]:.1f}" for j in range(t + 1)]
                print(f"done ({elapsed:.1f}s)  accs={seen_accs}")
        return self.results