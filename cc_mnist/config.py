"""
config.py
=========
All hyperparameters for the MNIST CC-vs-PC-vs-MLP experiment live here as
plain dataclasses, so a whole experiment configuration can be printed,
copied, swept, or serialized to JSON in one shot.

Where a number is taken (or adapted) directly from the source papers, the
docstring/comment says so explicitly, with a pointer to the section it
came from. Anything that is *not* in the papers (because the papers are
about a generative image model and we need a discriminative classifier)
is flagged as "(design choice)".
"""

from dataclasses import dataclass, field
from typing import List, Optional


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    dataset: str = "mnist"          # "mnist" or "digits" (sklearn fallback, 8x8, used when MNIST is unreachable)
    data_dir: str = "./data"
    val_fraction: float = 0.1       # carved out of the training set
    batch_size: int = 128
    normalize: str = "zero_one"     # "zero_one" or "standardize"
    flatten: bool = True
    seed: int = 0
    # If MNIST cannot be obtained (no internet / no torchvision / no local
    # cache), we transparently fall back to sklearn's bundled `load_digits`
    # (1797 8x8 images) purely so the *pipeline* can be smoke-tested. This
    # is loudly logged -- it is NOT a substitute for a real MNIST run.
    allow_fallback: bool = True


# --------------------------------------------------------------------------- #
# Network architecture
# --------------------------------------------------------------------------- #
@dataclass
class ArchConfig:
    # sizes[0] = input dim (e.g. 784), sizes[-1] = number of classes (10).
    # Everything in between is a hidden layer of the PC/CC hierarchy.
    sizes: List[int] = field(default_factory=lambda: [784, 256, 128, 10])
    hidden_activation: str = "tanh"          # matches ROW_SCALE_FROM_CHILD_DERIVATIVES options in the CC pseudocode
    use_skip: bool = True                    # gamma: skip-mix scalar (doc1 "Legend"), z2 -> output here
    gamma_init: float = 0.0
    gamma_max: float = 0.22                  # doc1 suggested schedule: keep gamma in ~[0.18, 0.25]
    use_lateral: bool = True                 # Lambda_k lateral quadratic prior (doc2 eq.1, eq.9) -- see README for
    lateral_init: float = 0.02               # the (documented) simplification used for non-grid hidden layers
    precision_init: float = 1.0              # Pi_k, isotropic precision per layer (doc2 Sec.2)
    weight_init_scale: float = 0.5           # Xavier-ish scale multiplier


# --------------------------------------------------------------------------- #
# PC inference (INFER_POSTERIORS, doc1 subroutine)
# --------------------------------------------------------------------------- #
@dataclass
class InferenceConfig:
    # doc1 suggests 28-34 for its (larger, generative, per-pixel) image model. Empirically,
    # on this discriminative classifier the relaxation converges far faster: sweeping
    # k_infer in {6,10,16,24} on the smoke architecture gave IDENTICAL validation accuracy
    # at every value (since this is a Jacobi/synchronous fixed point with a well-conditioned
    # preconditioner, it settles in just a few iterations for a network this shallow). 12 is
    # used as a safety margin for the deeper/wider real-MNIST architecture; try lowering it
    # (e.g. to 6-8) for a further ~2x speedup if accuracy holds, or raising it if you suspect
    # under-convergence (e.g. growing z_clip activity at very late iterations).
    k_infer: int = 12
    step_size: float = 0.5       # preconditioned step (divided by Pi_k+Lambda_k+1, see layers.py)
    mu_polyak: float = 0.5       # Polyak momentum for z updates (doc1 HYPER.mu_polyak)
    lambda_l1_z: float = 1e-3    # doc1 HYPER.lambda_L1_z ~= 1e-3
    z_clip: float = 8.0          # CLIP_OR_BOUND stabilization
    # If True: at TRAIN time, the clamped label's backprop gradient is injected into
    # EVERY relaxation iteration (genuine biclamped/bidirectional message passing, the
    # way classical supervised PC -- Whittington & Bogacz 2017 -- works). If False
    # (the default): the label is only used once, in a single backward pass AFTER the
    # hidden latents have settled top-down-only. We benchmarked both on the smoke
    # dataset and found True gives a much lower TRAINING loss but a WORSE validation
    # accuracy (84.6% vs 91.1% after 20 epochs) -- letting the relaxation see the
    # answer during settling lets it shortcut to a low loss for that specific batch
    # without the weights actually having to get better, which doesn't transfer to
    # test time (where the label is obviously unavailable). Left here as an opt-in for
    # experimentation, off by default. See README "Posterior inference" section.
    biclamp_during_relaxation: bool = False


# --------------------------------------------------------------------------- #
# CC-specific schedule (doc1 "Suggested schedules" + SCHEDULE legend)
# --------------------------------------------------------------------------- #
@dataclass
class ScheduleConfig:
    # Three-phase plan (the user's proposed pipeline, adopted as the default):
    #   Phase 1 "warmup"     : alpha = 0,                clarity off   -> behaves like vanilla discriminative PC
    #   Phase 2 "soft gating": alpha ramps 0.05 -> mid,   clarity off
    #   Phase 3 "full CC"    : alpha ramps to alpha_max,  clarity on & ramps
    warmup_epochs: int = 4
    soft_gate_epochs: int = 4          # length of the soft-gating ramp phase that follows warmup
    total_epochs: int = 16

    alpha_start: float = 0.05
    alpha_end: float = 0.60            # doc1: alpha_t: 0.05 -> 0.60 over epochs
    p_start: float = 1.05
    p_end: float = 1.35                # doc1: p_t: 1.05 -> 1.35

    clarity_warmup_epochs: int = 2       # clarity OFF for the first few epochs *of full_cc itself*; relative
                                          # to the full_cc phase's own length, NOT total_epochs -- see make_full_configs()
    lambda_diff_start: float = 0.0
    lambda_diff_end: float = 5e-3
    diffusion_order: int = 2            # low diffusion order (2-3) early, doc1
    clarity_every_n_batches: int = 5    # amortize the O(n^2)-ish diffusion cost (doc2 "Implementation notes")

    temperature_start: float = 1.0
    temperature_end: float = 0.6        # doc1: image temperature T_t: 1.0 -> 0.6

    gate_floor_min_fraction: float = 0.05  # the "+0.05*alpha_t" floor term in doc1 step 3.3


# --------------------------------------------------------------------------- #
# Optimization / regularization shared across PC and CC (doc1 HYPER)
# --------------------------------------------------------------------------- #
@dataclass
class OptimConfig:
    eta: float = 0.05            # HYPER.eta_l, per-layer learning rate (kept uniform here for simplicity)
    lambda_l1: float = 1e-5      # HYPER.lambda_L1 (weight sparsifier)
    lambda_l2: float = 1e-4      # HYPER.lambda_L2 (weight decay)
    # CC-only: extra L1 pressure scaled by (1 - G_l), i.e. proportional to how suppressed
    # a connection currently is. Without this, the gate only freezes adaptation (a
    # suppressed connection stays near its nonzero random init -- "amplitude-preserving",
    # not pruning); this term is what actually drives persistently-suppressed connections
    # toward true zero, so CC's gating shows up in a literal weight_sparsity metric and
    # not just in effective_rank/effective_connectivity. PCNetwork/MLP have no gate, so
    # this is simply unused for them. Set to 0.0 to recover the original (freeze-only,
    # not prune) gate behavior.
    lambda_l1_gated: float = 1e-3
    grad_clip: float = 5.0
    # MLP baseline uses plain SGD-with-momentum (NOT the Hebbian/PC machinery) -- it is the conventional control.
    mlp_lr: float = 0.1
    mlp_momentum: float = 0.9


# --------------------------------------------------------------------------- #
# Misc / experiment bookkeeping
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    name: str = "cc_vs_pc_vs_mlp_mnist"
    output_dir: str = "./outputs"
    seed: int = 0
    eval_every: int = 1                  # epochs between full eval+metric passes
    influence_consistency_batches: int = 40   # number of consecutive batches sampled for the stability metric
    tsne_n_samples: int = 1500
    smoke_test: bool = False             # drastically shrinks everything for a < 1 minute pipeline check


def make_smoke_configs():
    """A tiny, fast configuration used purely to verify the pipeline runs end to end."""
    data = DataConfig(dataset="digits", batch_size=32, val_fraction=0.15)
    arch = ArchConfig(sizes=[64, 32, 16, 10])
    infer = InferenceConfig(k_infer=6)
    sched = ScheduleConfig(
        warmup_epochs=1, soft_gate_epochs=1, total_epochs=3,
        clarity_warmup_epochs=1, clarity_every_n_batches=2,
    )
    optim = OptimConfig()
    exp = ExperimentConfig(
        name="smoke_test", smoke_test=True, influence_consistency_batches=8, tsne_n_samples=300,
    )
    return data, arch, infer, sched, optim, exp


def make_full_configs():
    """The real MNIST experiment configuration (requires MNIST + more wall-clock time)."""
    data = DataConfig(dataset="mnist", batch_size=128, val_fraction=0.1)
    arch = ArchConfig(sizes=[784, 256, 128, 10])
    infer = InferenceConfig()  # k_infer=12 default; see InferenceConfig's docstring/comment for the tuning notes
    # warmup=4, soft_gate=4 -> the "full_cc" phase is epochs [8,16), i.e. 8 epochs long.
    # clarity_warmup_epochs is *relative to the start of full_cc*, so this leaves clarity
    # active for the last 6 of those 8 epochs (previously the class default of 6 left
    # clarity active for only the last 2 of 16 epochs total -- barely enough to matter).
    sched = ScheduleConfig(warmup_epochs=4, soft_gate_epochs=4, total_epochs=16, clarity_warmup_epochs=2)
    optim = OptimConfig()
    exp = ExperimentConfig(name="cc_vs_pc_vs_mlp_mnist", smoke_test=False)
    return data, arch, infer, sched, optim, exp
