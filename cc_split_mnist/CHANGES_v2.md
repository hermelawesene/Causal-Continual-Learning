# v2 changes — drop these files over your existing project

## Changed files (9)
| file | what changed |
|---|---|
| `configs/default.yaml` | `cc.gate.g_max`, `cc.protect_strength`, `cc.lr_weights`, `pc.label_head` + `pc.ce_temperature`, `alpha_end` 0.60→0.85, `support_tau_quantile` 0.5→0.7, meta drift windows 1/10 |
| `core/gates.py` | **g_max cap** on the amplitude-preserving gate → fixes your task-4 NaN blowup (1+α·783 ≈ 470× multipliers on 784-wide layers) while preserving gate selectivity |
| `models/pc_network.py` | **CE label head** (`pc.label_head: ce`): ε_L = (onehot − softmax(μ_L/T))/T, the discriminative analogue of the pseudocode's crisp dL/dlogit-with-temperature; **label-error masking** for class-IL; masked softmax restricted to active classes (plain masked CE still zeroes old tasks via logit inflation — verified); `effective_label_pi()` = categorical Fisher diag for the influence recursion |
| `core/influence.py` | CE head support: label-layer curvature block = categorical Fisher (diagonal GN), no label lateral prior for observed logits |
| `training/trainer_pc.py` | **set_active_classes / masking** (PC *and* CC — fair comparison); **task-conditioned protection**: Ω = max past per-unit do-influence, rows suppressed by 1/(1+strength·relu(Ω−now)) — the neural analogue of the CCL paper's "freeze outside support" (Sec. 8.3), EWC-shaped but with do-influence replacing the Fisher; `prepare_task()` (pre-training influence, matching "estimate → support → update"); **meta wiring live** behind `cc.meta.enabled`: InfluenceTracker → LCB/UCB gate (normalized to max 1 so it stays selective without globally crushing lr) → per-unit multipliers, plus drift D_t printed at each refresh; CC-specific `cc.lr_weights` override |
| `training/continual.py` | calls `set_active_classes` (class-IL) and `prepare_task` before each task |
| `metrics/continual_metrics.py` | commutator probe applies the same class-IL label masking as real training (otherwise it measures interference training no longer has) |
| `training/trainer_backprop.py` | `loss.item()` warning fix; BP stays the *unprotected* fine-tuning baseline on purpose |
| `meta/hooks.py` | drift-detector warmup shortened for once-per-task cadence |

## What we verified on synthetic smoke tests
- No NaN anywhere (g_max fix), both scenarios, both heads, meta on/off.
- Class-IL exact-0.0 rows are gone for PC and CC (masking); remaining
  forgetting is trunk drift — the thing CC's protection targets.
- CC forgetting < PC forgetting (task-IL toy: 0.075 vs 0.213 avg forget;
  with `pc.label_head=ce`: 0.020). Expect the per-task-accuracy/forgetting
  trade to shift in CC's favor on real MNIST (15× more data per task).
- Drift statistic spikes at task boundaries (mechanism shifts), ~0 within.

## Run commands
```bash
# baseline grid (gaussian head)
python main.py --method all --scenario task_il
python main.py --method all --scenario class_il

# CE / softmax head (recommended to compare against gaussian)
python main.py --method cc --scenario class_il --override pc.label_head=ce
python main.py --method pc --scenario class_il --override pc.label_head=ce

# meta addon (LCB confidence gating + drift detection)
python main.py --method cc --scenario class_il --override cc.meta.enabled=true

# knobs to sweep next
#   cc.protect_strength: 4 / 8 / 16
#   cc.gate.alpha_end:   0.6 / 0.85
#   cc.influence.support_tau_quantile: 0.6 / 0.7 / 0.8
#   cc.meta.lcb_beta, cc.meta.explore_eps (only with meta.enabled=true)
```

## What to look at, in order
1. `leakage.png` — should drop vs your ~0.5–0.65 v1 numbers (locality ε_c)
2. `commutator_*.png` — off-diagonals CC < PC (small-commutator regime)
3. `support_overlap.png` — sparse = the confusion-graph condition of Thm 5
4. only then ACC/BWT — improvements should *follow* leakage going down;
   if leakage drops and BWT doesn't, that's a real finding about the gap
   between the theorem's assumptions and this architecture — record it.
