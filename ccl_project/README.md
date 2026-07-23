# Causal Continual Learning — Split MNIST

Implementation of the General Causal-Continual-Learning Theorem (Goertzel, 2025)
on the Split MNIST benchmark, comparing three methods:

| Method | Description |
|--------|-------------|
| `naive` | Standard MLP with backprop |
| `pc` | Predictive Coding network (modular backbone, no causal gates) |
| `pc_cc` | PC + Causal Coding (Jacobian gates + clarity regularizer) |

---

## Project Structure

```
ccl_project/
├── configs/
│   └── config.py          # All hyperparameters in one place
├── data/
│   └── dataset.py         # Split MNIST loader
├── models/
│   ├── naive_mlp.py       # Baseline MLP
│   ├── pc_network.py      # Predictive Coding network
│   └── pc_cc_network.py   # PC + Causal Coding network
├── trainers/
│   ├── base_trainer.py    # Shared training loop logic
│   ├── naive_trainer.py   # Trainer for naive MLP
│   ├── pc_trainer.py      # Trainer for PC network
│   └── cc_trainer.py      # Trainer for PC+CC (with gates + clarity)
├── visualization/
│   └── plots.py           # All plotting functions
├── outputs/               # Auto-created: saved plots + results
├── run_experiment.py      # Main entry point
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone / download this folder, then:
cd ccl_project

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the experiment
python run_experiment.py
```

Results and plots are saved to `outputs/`.

---

## Key Configuration (`configs/config.py`)

```python
N_EPOCHS   = 5       # epochs per task
LR         = 1e-3    # Adam learning rate
N_MODULES  = 4       # causal modules in backbone
CLARITY_W  = 0.005   # weight of clarity (cross-module) penalty
GATE_FREQ  = 50      # steps between Jacobian re-estimation
```

---

## Theory Summary

The CCL theorem says forgetting is bounded by:

```
|L_i(θ_after_j) - L_i(θ_before_j)| ≤ K · ε_comm

where ε_comm = C(ε_h · G_max + ε_g · H_max)
  ε_g = gradient norm outside a context's support modules
  ε_h = cross-module Hessian block norm
```

Causal coding suppresses both ε_g (via Jacobian gates) and ε_h (via clarity penalty),
making ε_comm small, which bounds forgetting.
