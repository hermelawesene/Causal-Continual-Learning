"""Entry point.

Examples:
    python main.py --method cc --scenario task_il
    python main.py --method pc --scenario class_il
    python main.py --method all --scenario task_il
    python main.py --method cc --scenario class_il --override cc.meta.enabled=true
    python main.py --method cc --scenario task_il --override pc.label_head=ce
"""
from __future__ import annotations

import argparse
import copy
import os
import random

import numpy as np
import torch
import yaml

from data.split_mnist import SplitMNIST
from training.continual import run_continual
from training.trainer_backprop import BackpropTrainer
from training.trainer_pc import CCTrainer, PCTrainer

TRAINERS = {"bp": BackpropTrainer, "pc": PCTrainer, "cc": CCTrainer}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def apply_overrides(cfg, overrides):
    for ov in overrides:
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        old = node.get(parts[-1])
        node[parts[-1]] = yaml.safe_load(val)
        print(f"override {key}: {old} -> {node[parts[-1]]}")
    return cfg


def build_trainer(method, cfg, data, device):
    n_heads = data.num_tasks if cfg["scenario"] == "task_il" else 1
    return TRAINERS[method](cfg, data.label_dim(), n_heads, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["bp", "pc", "cc", "all"], default="cc")
    ap.add_argument("--scenario", choices=["task_il", "class_il"], default=None)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__),
                                                     "configs", "default.yaml"))
    ap.add_argument("--override", nargs="*", default=[],
                    help="dot.path=value config overrides")
    ap.add_argument("--tag", default="", help="suffix for the results directory")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.scenario:
        cfg["scenario"] = args.scenario
    cfg = apply_overrides(cfg, args.override)

    set_seed(cfg["seed"])
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if cfg["device"] == "auto" else cfg["device"]
    print(f"device={device}, scenario={cfg['scenario']}")

    data = SplitMNIST(cfg)
    methods = ["bp", "pc", "cc"] if args.method == "all" else [args.method]

    results = {}
    for m in methods:
        set_seed(cfg["seed"])
        run_dir = os.path.join(cfg["results_dir"],
                               f"{m}_{cfg['scenario']}{('_' + args.tag) if args.tag else ''}")
        trainer = build_trainer(m, copy.deepcopy(cfg), data, device)
        out = run_continual(trainer, data, cfg, run_dir)
        results[m] = np.array(out["R"])

    if len(results) > 1:
        from plotting import plots
        cmp_dir = os.path.join(cfg["results_dir"],
                               f"comparison_{cfg['scenario']}"
                               f"{('_' + args.tag) if args.tag else ''}")
        rand_acc = 1.0 / data.label_dim()
        plots.plot_method_comparison(results, rand_acc, cmp_dir)
        plots.plot_curves(results, cmp_dir)
        print(f"comparison plots -> {cmp_dir}")


if __name__ == "__main__":
    main()
