
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs       import CFG
from data          import get_split_mnist
from trainers      import NaiveTrainer, PCTrainer, CCTrainer
from visualization import plot_comparison, plot_gates, plot_per_task_accuracy


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

print("\n  Loading Split MNIST ...")
train_loaders, test_loaders, task_names = get_split_mnist()
print("METHOD 3 — PC + Causal Coding  (Jacobian gates + clarity penalty)")
set_seed(CFG.SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cc_t = CCTrainer(train_loaders, test_loaders, task_names, device)
res_cc = cc_t.run(verbose=True)
res_cc.print_summary()

if cc_t.gate_history_l1:
        plot_gates(cc_t.gate_history_l1, cc_t.gate_history_l2, task_names,
                   output_dir=CFG.OUTPUT_DIR,
                   save=CFG.SAVE_FIG, show=CFG.SHOW_FIG)
   
