# Causal Coding vs Predictive Coding vs MLP on MNIST

A from-scratch, pure-NumPy implementation comparing three ways of training
a classifier on MNIST:

- **MLP** — a conventional backprop multilayer perceptron. The control condition.
- **PC** — "vanilla" discriminative Predictive Coding: hidden representations come from
  iterative free-energy relaxation instead of one feedforward pass, and weights are
  updated by a local Hebbian rule. No causal-coding machinery.
- **CC** — Causal Coding: PC plus the naturalized influence, amplitude-preserving gate,
  and diffusion-based "clarity" term from the source pseudocode, driven by a three-phase
  training schedule.

This README documents *how* the generative image-modeling theory in the three uploaded
papers was re-mapped onto a discriminative classification setting, what was kept
faithful to the letter of the pseudocode, and what had to be simplified or adapted —
so that every modeling choice in `models.py` can be traced back to a specific
equation or pseudocode step.

Source documents referenced below:
- **doc1** = "Causal Coding (CC) Training Loop: Fully Commented Pseudocode"
- **doc2** = "Do-Influence and Natural-Gradient Surrogates for Causal Coding" (the math formalism)
- **doc3** = "Causal-Continual-Learning-Theorem-v1.pdf" (only used here for its eq.22-24
  description of standard PC's inference/update equations — the continual-learning
  theorem itself is out of scope for this experiment)


## 1. The discriminative re-mapping

doc1's pseudocode is a *generative* model of images: a clamped top latent generates an
image at the bottom through a chain of hierarchical latents, and the per-pixel image
loss (`IMAGE_LOSS_LOGIT_GRAD`) is injected at the leaf.

To turn this into an MNIST *classifier* with minimal surgery, the hierarchy is used in
exactly the same direction, with the roles of "clamped top" and "generated leaf"
reassigned:

```
z^L = x        the input image.            CLAMPED.    (doc1's role for the image, here it's the "given")
z^{L-1} .. z^1  tanh hidden latents.        RELAXED.    (the only iteratively-settled free latents)
z^0 = y        class logits.                "GENERATED" leaf, crisp softmax-CE loss.
```

So the input plays the role doc1's image played (the thing that's given and clamped),
and the label plays the role doc1's image-leaf played (the thing with a crisp,
logit-space loss attached to it) — just one level removed, since y is categorical
rather than a per-pixel field. `label_loss_grad` in `models.py` is the direct
discriminative analog of doc1's `IMAGE_LOSS_LOGIT_GRAD`: a crisp gradient computed in
**logit space** (`dCE/da = (softmax(a/T) - y_onehot)/T`), not in probability space —
matching doc1's explicit emphasis on logit-space crispness.

**How the label participates, by default.** The hidden latents z^1..z^{L-1} are
iteratively relaxed via the free-energy fixed point (`infer()` in `models.py`) using
*only* top-down information from x. The label z^0=y is, by default, used exactly
once — after the hidden latents have already settled — via a single standard backprop
pass (`backprop_deltas_from_label`) that computes the weight gradients. It's also
*possible* to clamp y into every relaxation iteration too (genuine bidirectional
"biclamped" settling, the way classical supervised PC works), via
`InferenceConfig.biclamp_during_relaxation=True` — but we benchmarked it and it
measurably hurts validation accuracy (see §2 and §6b below for the numbers and the
reasoning), so it's off by default. See §2 for the full, plain-language walkthrough of
exactly what `infer()` does and why.


## 2. Posterior inference, in plain language

Forget the equations for a second. Here's what's actually happening on every
training batch:

1. **Look at the image once, get a rough guess.** `feedforward_init` runs one
   ordinary forward pass (input -> hidden1 -> hidden2) to get a starting point
   for the hidden layers. This is just a sensible initial guess, not "the
   answer" yet.

2. **Let the hidden layers argue with each other for a few rounds.** Picture
   each hidden layer as having an opinion about what it should look like,
   based on two things:
   - what the layer *above* it predicts it should be (a "top-down" guess —
     e.g. hidden layer 2 has an opinion about what hidden layer 1 should look
     like, computed by literally running hidden layer 2 through the weights
     that connect them);
   - whether the layer *below* it is happy with the prediction *it's* making
     (a "bottom-up" complaint — if hidden layer 1's own prediction error is
     large, that's a signal that hidden layer 2 should adjust).

   Each round, every hidden layer nudges itself a little to reduce its own
   disagreement with its neighbors (`infer()`'s main loop). This repeats
   `k_infer` times (12 by default) until things settle down. This is the
   "inference" or "relaxation" step — doc1 calls it `INFER_POSTERIORS`.
   Nothing about the label is involved here by default; it's purely the
   image "talking" up through the hidden layers.

3. **Once everything's settled, read off a prediction.** `forward_logits`
   takes the now-settled hidden layers and does one more forward pass to
   produce class logits, exactly like a normal feedforward network would.

4. **Compare the prediction to the true label, and backpropagate that error
   *once*** (only at train time, since only then do we know the true label)
   **to update every weight.** This is `backprop_deltas_from_label`: ordinary
   backprop, the same algorithm an MLP uses, just applied to the network
   *after* it's done its multi-round settling instead of after a single
   forward pass.

So: the only real difference between PC and a normal MLP is step 2 — instead
of computing each hidden layer in one shot, PC lets the hidden layers iterate
toward a value that's mutually consistent with their neighbors before reading
off a prediction. Steps 3-4 are completely ordinary.

**Why bother with step 2 at all, if it's not even using the label?** Because
this loop is what doc1's "Causal Coding" machinery (section 4 below) operates
on: by letting the hidden layers settle into a state that's genuinely
predictive of each other (not just whatever a single forward pass spits out),
the per-connection "how much does this weight actually matter" signal (`M_l`,
section 4) becomes a much more meaningful, stable quantity to gate on. CC is
PC plus that extra gating step.

### Why isn't the label *also* clamped during this settling?

This is the natural follow-up question, and the honest answer is: **we tried
it, and it made validation accuracy *worse*.** Classical supervised
predictive-coding networks (Whittington & Bogacz 2017) do clamp the label at
the leaf and let it influence the hidden layers' settling too, not just the
final backward pass — and we built that option (it's the
`InferenceConfig.biclamp_during_relaxation` flag, off by default). When we
turned it on, training loss dropped dramatically faster, but validation
accuracy got *worse* (84.6% vs 91.1% after 20 epochs on the smoke
architecture). The reason makes sense in hindsight: if the hidden layers are
allowed to see the answer while they're still settling, they can cheat their
way to a low loss *for that specific batch* by leaning on the relaxation
process itself, rather than the weights actually getting better — and that
shortcut obviously isn't available at test time, where the label is unknown.
So by default the label is only used once, after the hidden layers have
already committed to a settled state based on the image alone. You can flip
the flag back on if you want to experiment with it yourself.

### The formal version, for reference

Implements doc1's `INFER_POSTERIORS` / doc3's eq.22-23 synchronous (Jacobi-style)
fixed-point iteration. For each free latent k = 1..L-1, every iteration computes:

```
err_self_k  = Pi_k * (z^k - g_k(z^{k+1}))          # own top-down prediction error
bottom_up_k = J_{k-1}^T @ err_self_{k-1}            # error pulled down from the layer below (0 if k=1)
lateral_k   = Lambda_k * z^k                        # quadratic lateral prior
dF/dz^k     = err_self_k + lateral_k - bottom_up_k
z^k        <- z^k - step_size * dF/dz^k / (Pi_k + Lambda_k + 1) + momentum + L1 shrinkage
```

Initialization is a single ordinary feedforward pass (`feedforward_init`) rather than
zeros — a free, sensible starting point for the relaxation that doc1 doesn't forbid and
that markedly speeds convergence within a fixed `k_infer` budget.


## 3. Weight update: standard backprop, applied after settling (doc1 step 3.1 / doc3 eq.24)

After the hidden latents settle, `backprop_deltas_from_label` does exactly what an
MLP's backward pass does — propagate the crisp label-loss gradient upward through the
*transposed weight Jacobians*, correctly multiplying by each layer's own activation
derivative at every step:

```
delta_pre_0 = g_y                                                        (act_0 = identity)
delta_z_p   = sum over layers with parent p of  scale_l * (delta_pre_{child(l)} @ W_l)
delta_pre_p = delta_z_p * act_p'(pre_p)                                   for p = 1..L-1
dW_l        = mean_batch( outer(delta_pre_{child(l)}, z^{parent(l)}) )
```

This is doc1's own description of this step — "pass `g_img` ... exactly where PC would
backprop" — and is what makes vanilla `PCNetwork` equivalent to doc3's "standard PC"
(Sec 2.4 / 3.5 / 6).

**A bug we found and fixed:** an earlier version of this code used `delta_z` directly
in the outer product instead of `delta_pre` (i.e. it skipped the `act_p'(pre_p)`
multiplication for every hidden layer). Since the output layer's activation is
identity (derivative 1), this bug was invisible there, but it silently gave every
*hidden* layer a systematically wrong gradient — almost certainly the main reason PC
and CC plateaued well below the MLP baseline in the first real-MNIST run. It's fixed
now; see the README's note in `models.py`'s `hebbian_grads` docstring.

There is no autodiff anywhere in this codebase — every gradient above is closed-form,
by design (see §7 below).


## 4. Causal Coding additions (doc1 steps 3.2–3.4)

`CCNetwork` adds three things on top of `PCNetwork`, applied as a uniform "for each
layer" loop (the chain weights W0..W_{L-1} *and* the optional skip connection are
all just entries in `self.layers`, so every CC mechanism below treats them identically):

**Naturalized influence `M_l`** (step 3.2):
```
row_scale = mean(|child_der|) / (mean(child_der^2) + eps)     # per output unit
col_scale = 1 / sqrt(mean(parent_pre^2) + eps)                  # per input unit
M_l       = row_scale[:, None] * |W_l| * col_scale[None, :]
```
`child_der` is the activation derivative of whatever sits at the layer's child index:
`tanh'` for the hidden chain, and — since our leaf is now a softmax classifier rather
than doc1's sigmoid-per-pixel image — `softmax curvature p(1-p)/T` for any layer
feeding the output (the direct categorical analog of doc1's own sigmoid-derivative
choice for its leaf).

**Amplitude-preserving gate `G_l`** (step 3.3), with a stop-gradient (it only ever
scales the already-computed Hebbian gradient, never participates in computing it):
```
R_l = NORMALIZE_INPUTWISE(|M_l|^p_t)
G_l = max( 1 + alpha_t*(n_in*R_l - 1),  1 - alpha_t + gate_floor_fraction*alpha_t )
dW_l <- G_l * dW_l
```

**A conceptual trap worth flagging explicitly:** this gate multiplies the *gradient*,
not the weight. A suppressed connection (`G_l < 1`) just gets a *smaller update* — it
freezes near its random initialization, which is not near zero. That's exactly what
"amplitude-preserving" means in the name. So by itself, this gate does **not** produce
literal weight-magnitude sparsity, even though it's doing real, useful work (you can
see it in `effective_rank`/`effective_connectivity`, or in the gate's own
`gate_std`/`gate_suppressed_frac` history). If you measure plain "% of weights near
zero" (`weight_sparsity` in `metrics.py`), the gate alone won't move that number much.

**Gated pruning (our addition, not in doc1):** to get genuine pruning-style sparsity
out of the gate, we added one more term, scaled by how suppressed a connection
currently is:
```
prune_l = lambda_l1_gated * max(1 - G_l, 0) * sign(W_l)
dW_l   <- G_l * dW_l + prune_l
```
This is plain L1 shrinkage, but its strength is *proportional to the gate's own
suppression signal* — connections CC has identified as low-influence get pushed
toward true zero over time, instead of just having their adaptation frozen.
`OptimConfig.lambda_l1_gated` (default `1e-3`) controls its strength; set it to `0.0`
to recover the original (freeze-only, non-pruning) gate. On the smoke architecture
this raised CC's `overall_sparsity` from being only marginally above PC's (~24%
relative gap) to clearly above it (~46% relative gap), with no accuracy cost.

**Clarity `D_l`** (step 3.4), throttled to every `clarity_every_n_batches` batches for
cost: build a block adjacency matrix `A` out of the row-normalized `M_l` for every
layer, diffuse it (`K = sum_{i=1..order} A^i`), and add back the *indirect* (multi-hop,
"overshadowed") influence that the direct one-hop connection doesn't already explain:
```
P_l   = NORMALIZE_INPUTWISE(M_l)                       # NOT raised to p_t, per doc1 3.4
os_l  = max(K[child,parent] - A[child,parent] - delta, 0)
D_l   = os_l * P_l * sign(W_l)
dW_l <- dW_l + lambda_diff_t * D_l
```


## 5. The three-phase training pipeline

The schedule (`schedule.py`, `CCSchedule`) implements the pipeline the user specified
as the default, since it's a sensible and principled way to introduce CC's machinery
gradually:

| Phase | What's active | Why |
|---|---|---|
| **1. Warmup** (`alpha_t=0`, clarity off) | Pure discriminative PC | Lets the relaxation dynamics and the backward-propagated Hebbian signal stabilize *before* anything starts reweighting it. Gating an untrained, noisy gradient signal would just amplify noise. |
| **2. Soft gating** (`alpha_t` ramps to a small value) | Gate only | Eases the gate in slowly so early-phase weight reorganization doesn't get destabilized by a sudden hard reweighting. |
| **3. Full CC** (`alpha_t -> alpha_end`, `p_t -> p_end`, clarity turns on after a short delay and ramps) | Gate + clarity | By this point the influence estimates `M_l` are themselves computed from a reasonably-converged network, so the gate's reweighting and the clarity term's "indirect influence" correction are acting on a meaningful signal rather than noise. |

Temperature `T_t` (1.0 → 0.6) anneals throughout *all three phases for both PC and
CC* — in doc1 it's set at the generic top-level training step (i.e. it's a property of
the crisp readout loss generally), not something specific to the CC machinery, so
`PCNetwork.train_step` also takes a `T` argument and the same schedule's `T_t` is used
for it. The only things gated to CC specifically are `alpha_t`, `p_t`, `clarity_on`,
and `lambda_diff_t`.

We deliberately did *not* implement a loss-plateau-triggered phase transition (e.g.
"advance to the next phase once validation loss stalls") even though it's a reasonable
alternative — fixed epoch boundaries make the three runs (MLP/PC/CC) directly
comparable under an identical, non-adaptive training budget, which matters more here
than getting the single best CC run.


## 6. Documented simplifications

Being candid about where this departs from an exact reading of doc1/doc2:

- **Lateral term `Lambda_k`** is a simplified **diagonal**, per-unit term (learnable in
  principle, currently fixed), rather than doc2's eq.1/eq.9 pixel-grid Laplacian.
  Hidden units in a classifier have no natural grid topology for a Laplacian to act on,
  so a diagonal quadratic prior (effectively an L2-towards-zero pull, independent of
  neighboring units) is the natural reduction and is flagged here rather than
  silently assumed.
- **Precision `Pi_k`** is a fixed scalar per layer, not learned and not a full
  precision matrix. doc1's own `M_l` formula doesn't reference `Pi` at all (it's the
  "tractable surrogate" version of doc2's exact `(Pi+Lambda)^-1 Pi J` do-influence), so
  this only affects the *inference* dynamics, not the CC gate/clarity machinery, and
  we follow doc1's own simplified-surrogate path there rather than doc2's exact one.
- **Row/col scale damping (`row_scale_from_child_derivatives`, `col_scale_from_parent`
  in `layers.py`) uses a *layer-relative* epsilon, not doc1's literal absolute "+eps".**
  A near-constant-zero input (an MNIST border pixel that's background in nearly every
  image) or a saturated tanh unit has near-zero variance/derivative; an absolute
  epsilon floor (e.g. 1e-6) makes `1/sqrt(var+eps)` blow up for exactly those
  units — i.e. the *most useless* connections would get reported as the *most
  influential*, the opposite of the gate's intent. We instead floor each layer's
  variance/curvature at a small fraction of that *layer's own* typical scale, which is
  the same fix natural-gradient/K-FAC approximations use for the identical pathology
  (Martens & Grosse). This was likely muting CC's gate in the first real-MNIST run
  (gate values stayed within 1.4% of 1.0 across all 16 epochs); after the fix, the
  gate shows real per-connection redistribution (std growing past 0.7 by the end of
  training on the smoke architecture).
- **Skip connection** `z^2 -> y` (gated by scalar `gamma`, capped at `gamma_max`) is
  the direct analog of doc1's `z2 -> x` skip, just generalized into the uniform layer
  list so every CC mechanism (Hebbian update, gate, clarity, regularization) treats it
  identically to the main chain layers rather than needing special-cased code.
- **The label is only used once, after the hidden layers settle**, by default (see §2
  above) — clamping it into the relaxation itself is implemented and available
  (`InferenceConfig.biclamp_during_relaxation=True`) but is off by default since it
  measurably hurt validation accuracy in our tests.


## 6b. Bugs found and fixed after the first real-MNIST run

The first full run (real MNIST, 16 epochs) surfaced four separate issues, all now fixed:

1. **~110-minute runtime.** `infer()`'s bottom-up term was building an explicit
   `(batch, n_in, n_out)` tensor every single relaxation iteration when it's
   mathematically just a matrix multiply. Fixed by computing it directly as
   `(der * err) @ W` with no intermediate tensor. Combined with reducing `k_infer`
   from 24 to 12 (see §2's empirical finding that accuracy was identical down to
   `k_infer=6` on the smoke architecture), a full 16-epoch real-MNIST run should now
   take on the order of 10-15 minutes total for all three models, not ~2 hours.
2. **Missing activation-derivative factor in the weight gradient** (§3 above) — likely
   the dominant reason PC/CC plateaued around 93% instead of approaching the MLP's
   97.5%. Fixed.
3. **Clarity barely turned on.** The default schedule's `clarity_warmup_epochs=6` was
   being measured from the *start of `full_cc`*, but with `warmup=4, soft_gate=4,
   total=16` the `full_cc` phase is only 8 epochs long — so clarity was active for
   just the last 2 of 16 epochs. Fixed by setting `clarity_warmup_epochs=2` in
   `make_full_configs()`, leaving clarity active for 6 of those 8 epochs.
4. **Gate values stuck within ~1% of 1.0** — the absolute-epsilon damping issue
   described above. Fixed via layer-relative damping.

After all four fixes, on the smoke architecture/dataset (20 epochs, otherwise default
schedule) CC clearly outperforms PC for the first time: 93.5% vs 91.2% validation
accuracy, with the gate showing real redistribution (std rising to ~0.78) and clarity
active for the back half of training. This obviously isn't real-MNIST-scale evidence,
but it's the qualitative result the gating/clarity machinery is supposed to produce,
which it wasn't before these fixes.


## 7. Why pure NumPy, no autodiff

This sandbox has no internet access and no torch/torchvision installed. That turned
out to be a non-issue: every gradient in this codebase is closed-form (the free energy
is quadratic in each latent, and the crisp readout loss is the standard softmax-CE
chain rule), so no autodiff framework is needed at all — which, if anything, is *more*
faithful to doc1's own architecture-agnostic pseudocode style than wrapping things in
a deep learning framework would have been.


## 8. Metrics

All five families the user asked for are implemented in `metrics.py`:

1. **Performance** — test accuracy/loss, confusion matrix, per-class accuracy.
2. **Sparsity / modularity** — % near-zero weights, effective connectivity (active
   inputs per output unit), per weight matrix. **Caveat:** MLP, PC, and CC share the
   same nominal `lambda_l1`/`lambda_l2` coefficients, but MLP's optimizer (momentum
   SGD, `mlp_lr=0.1`, `momentum=0.9`) amplifies their steady-state pull toward zero by
   roughly `1/(1-momentum) = 10x` relative to PC/CC's bare `eta=0.05` with no
   momentum. So MLP showing much higher `weight_sparsity` than PC/CC is largely an
   optimizer artifact, not evidence about which learning rule produces sparser
   weights -- treat PC-vs-CC comparisons on this metric as meaningful, MLP-vs-(PC/CC)
   comparisons on this specific metric as confounded.
3. **Entanglement** — per-neuron count of how many classes drive it to within 50% of
   its own peak class-conditional response (low = selective, high = entangled);
   discretized neuron-label mutual information; pairwise neuron-neuron activation
   correlation.
4. **Influence consistency** — within the *same* trained CC model (frozen weights),
   run `compute_M_no_update` over many consecutive minibatches and compute the mean
   batch-to-batch cosine similarity of (a) the raw `|Hebbian dW|` sequence and (b) the
   `M_l` sequence, per layer. This is the most direct possible test of "does CC assign
   more *stable* importance than the raw gradient magnitude would" — same network,
   same data, two different importance signals computed side by side. (A simpler
   cross-model raw-gradient-consistency overview is also computed for context.)
5. **Representation structure** — t-SNE of the first hidden layer's activations,
   colored by class, plus a silhouette score of the 2D embedding against true labels.

**Extras added beyond the requested list** (all justified inline in `metrics.py`):
expected calibration error (ECE), effective weight-matrix rank (singular-value
entropy — lower means a weight matrix is using fewer effective directions, i.e. more
structured/modular), and per-neuron entanglement histograms for a fuller picture than
the single summary scalar.

**Deliberately not added:** a cross-class "gradient interference / Lie-bracket
commutator" metric, even though it would be a natural extension of doc3's theorem.
That metric is squarely about *continual learning* (how much does learning task B
disturb task A), which the user explicitly scoped out of this experiment. It's
mentioned here only as a natural next step if a sequential/continual variant of this
experiment is run later.


## 9. Running this for real

This sandbox has no internet access, so `data.py` cannot download real MNIST here.
`run_experiment.py --smoke` uses sklearn's bundled 8x8 `digits` dataset (1797 samples)
purely to verify the *pipeline* runs end to end — **it is not a substitute for a real
MNIST run** and the relative MLP/PC/CC numbers it produces should not be read as a
real result (the smoke architecture is also far smaller: `[64, 32, 16, 10]` instead of
MNIST's `[784, 256, 128, 10]`).

To run the real experiment on your own machine, `data.py` already supports three
strategies (it tries them in order automatically — no flags needed beyond
`--dataset mnist`, which is also the default):

1. `sklearn.datasets.fetch_openml('mnist_784')` — works if you have internet and
   scikit-learn (most common case).
2. Raw IDX files — drop the four classic
   `{train,t10k}-{images,labels}-idx{3,1}-ubyte.gz` files into `./data/` (e.g.
   downloaded once from any mirror) and it'll load from disk with no network needed
   on subsequent runs.
3. `torchvision.datasets.MNIST(download=True)` — used if torch/torchvision are
   installed.

```bash
pip install -r requirements.txt
python run_experiment.py --smoke                  # ~1s pipeline sanity check
python run_experiment.py                           # full MNIST run (needs MNIST access)
python run_experiment.py --epochs 20 --seed 1       # override epoch budget / seed
```

**Expected runtime:** benchmarked directly on the real `[784, 256, 128, 10]`
architecture with `k_infer=12` (the current default), PC runs at roughly 17s/epoch and
CC (full-gating + clarity phase) at roughly 26s/epoch on a single CPU core -- so the
full 16-epoch default run should take on the order of 10-15 minutes total for all
three models combined, not the ~110 minutes the pre-fix code took. If you want it
faster still, `InferenceConfig.k_infer` is the main lever: we found *zero* accuracy
difference sweeping it from 6 to 24 on the smoke architecture, so try lowering it
(e.g. to 6-8) first; raise it again if you see signs of under-convergence on the full
network.

Outputs land in `./outputs/` (`--output_dir` to change): `figures/*.png` for every
plot, `report.json` for the numeric summary, plus a console log of every epoch.


## 10. File map

```
config.py     dataclasses for every hyperparameter; make_smoke_configs() / make_full_configs()
utils.py      softmax/one-hot/cross-entropy/accuracy, the minibatch iterator, misc helpers
data.py       MNIST loading (3 real strategies + the digits smoke-test fallback)
layers.py     activation fns, weight init, ROW_SCALE/COL_SCALE/NORMALIZE_INPUTWISE (doc1 1:1)
schedule.py   CCSchedule: the 3-phase warmup -> soft-gate -> full-CC pipeline
models.py     MLP, BaseHierarchical (shared PC/CC machinery), PCNetwork, CCNetwork
metrics.py    all 5 metric families + the extras
plotting.py   every figure-saving function (matplotlib/seaborn, headless Agg backend)
run_experiment.py   orchestrates training + evaluation + plotting + report.json
```
