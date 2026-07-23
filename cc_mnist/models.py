"""
models.py
=========
Three models, sharing as much code as honestly possible so the *only*
differences between them are the ones the papers actually describe:

  MLP        : conventional backprop MLP. The control condition.

  PCNetwork  : "discriminative predictive coding". The input is clamped at the
               top (z^L = x); the label is clamped at the leaf (z^0 = y) at
               TRAIN time. Only the hidden latents z^1..z^{L-1} are
               iteratively relaxed (doc1 INFER_POSTERIORS / doc3 eq.22-23).
               By default the label only enters via a single backward pass
               AFTER the hidden latents settle (top-down-only relaxation);
               optionally (`InferenceConfig.biclamp_during_relaxation=True`)
               it can instead be injected into every relaxation iteration,
               for genuine bidirectional message passing the way classical
               supervised PC works (Whittington & Bogacz 2017) -- we
               benchmarked this and found it hurts validation accuracy (see
               README), so it's off by default. Every weight update is a
               single backprop-style outer product -- doc1's own phrase,
               "exactly where PC would backprop" -- now correctly including
               each layer's own activation derivative (previously a bug).

  CCNetwork  : PCNetwork + the naturalized influence M_l, the
               amplitude-preserving stop-grad gate G_l, and the
               diffusion-based clarity term D_l (doc1 steps 3.2-3.4),
               driven by the doc1 SCHEDULE.

See README.md, section "Discriminative re-mapping of the generative CC
pseudocode" and section "Posterior inference, in plain language", for the
full derivation/explanation of:

    z^L = x        the input image.            CLAMPED (always).
    z^{L-1}..z^1   tanh hidden latents.         RELAXED (always).
    z^0 = y        class logits/probabilities.  CLAMPED at train time only;
                                                  read out via a single
                                                  feedforward pass at test
                                                  time, since nothing clamps
                                                  it then.
"""
from dataclasses import asdict
import numpy as np

from layers import (
    ACTIVATIONS, init_layer, row_scale_from_child_derivatives,
    col_scale_from_parent, normalize_inputwise, stop_gradient,
)
from utils import softmax, one_hot, cross_entropy, accuracy


# --------------------------------------------------------------------------- #
# Baseline: conventional backprop MLP (the control condition)
# --------------------------------------------------------------------------- #
class MLP:
    def __init__(self, arch_cfg, n_classes, input_dim, optim_cfg, seed=0):
        self.sizes = list(arch_cfg.sizes)
        self.sizes[0], self.sizes[-1] = input_dim, n_classes
        self.optim = optim_cfg
        self.rng = np.random.RandomState(seed)
        self.Ws, self.bs = [], []
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            W, b = init_layer(n_out, n_in, arch_cfg.weight_init_scale, self.rng)
            self.Ws.append(W)
            self.bs.append(b)
        self.vW = [np.zeros_like(W) for W in self.Ws]
        self.vb = [np.zeros_like(b) for b in self.bs]
        self.n_layers = len(self.Ws)
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def forward(self, X, return_hidden=False):
        a = X
        hiddens = [X]
        pre_acts = []
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            pre = a @ W.T + b
            pre_acts.append(pre)
            if i < self.n_layers - 1:
                a = np.tanh(pre)
            else:
                a = pre  # logits
            hiddens.append(a)
        probs = softmax(a)
        if return_hidden:
            return probs, hiddens, pre_acts
        return probs

    def get_hidden_activations(self, X, layer_idx=1):
        """layer_idx counts hidden layers from the input (1 = first hidden layer)."""
        _, hiddens, _ = self.forward(X, return_hidden=True)
        return hiddens[layer_idx]

    def train_step(self, X, y):
        n = X.shape[0]
        probs, hiddens, pre_acts = self.forward(X, return_hidden=True)
        y1h = one_hot(y, self.sizes[-1])
        loss = cross_entropy(probs, y)

        delta = (probs - y1h) / n  # dL/dlogits, standard softmax+CE
        grads_W, grads_b = [None] * self.n_layers, [None] * self.n_layers
        for i in reversed(range(self.n_layers)):
            pre_in = hiddens[i]
            grads_W[i] = delta.T @ pre_in
            grads_b[i] = delta.sum(axis=0)
            if i > 0:
                d_hidden = delta @ self.Ws[i]
                t = np.tanh(pre_acts[i - 1])
                delta = d_hidden * (1 - t * t)

        for i in range(self.n_layers):
            gW = grads_W[i] + self.optim.lambda_l2 * self.Ws[i] + self.optim.lambda_l1 * np.sign(self.Ws[i])
            gW = np.clip(gW, -self.optim.grad_clip, self.optim.grad_clip)
            gb = np.clip(grads_b[i], -self.optim.grad_clip, self.optim.grad_clip)
            self.vW[i] = self.optim.mlp_momentum * self.vW[i] - self.optim.mlp_lr * gW
            self.vb[i] = self.optim.mlp_momentum * self.vb[i] - self.optim.mlp_lr * gb
            self.Ws[i] += self.vW[i]
            self.bs[i] += self.vb[i]
        return loss

    def predict(self, X):
        return self.forward(X)

    def get_weight_matrices(self):
        return list(self.Ws)

    def compute_grads_no_update(self, X, y):
        """Same gradient computation as train_step, but does not touch the weights.
        Used by the influence-consistency metric to get a comparable 'raw backprop
        gradient magnitude' stream for the MLP control condition."""
        n = X.shape[0]
        probs, hiddens, pre_acts = self.forward(X, return_hidden=True)
        y1h = one_hot(y, self.sizes[-1])
        delta = (probs - y1h) / n
        grads_W = [None] * self.n_layers
        for i in reversed(range(self.n_layers)):
            grads_W[i] = delta.T @ hiddens[i]
            if i > 0:
                d_hidden = delta @ self.Ws[i]
                t = np.tanh(pre_acts[i - 1])
                delta = d_hidden * (1 - t * t)
        return {f"layer{i}": gW for i, gW in enumerate(grads_W)}


# --------------------------------------------------------------------------- #
# Shared PC/CC hierarchy
# --------------------------------------------------------------------------- #
class BaseHierarchical:
    def __init__(self, arch_cfg, infer_cfg, n_classes, input_dim, seed=0):
        sizes = list(arch_cfg.sizes)
        sizes[0], sizes[-1] = input_dim, n_classes
        # internal z-index convention: 0 = output/label (leaf), L = input image (clamped top)
        self.sizes_z = list(reversed(sizes))
        self.L = len(self.sizes_z) - 1
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.arch = arch_cfg
        self.infer_cfg = infer_cfg
        self.rng = np.random.RandomState(seed)
        self.hidden_act_name = arch_cfg.hidden_activation

        # --- main chain layers: W_k maps z^{k+1} (parent) -> z^k (child), k = 0..L-1
        self.layers = []  # each: dict(name, W, b, parent, child, act_name, is_skip)
        for k in range(self.L):
            act_name = "identity" if k == 0 else self.hidden_act_name
            W, b = init_layer(self.sizes_z[k], self.sizes_z[k + 1], arch_cfg.weight_init_scale, self.rng)
            self.layers.append(dict(name=f"W{k}", W=W, b=b, parent=k + 1, child=k, act_name=act_name, is_skip=False))

        # --- optional skip connection: z^2 -> y (child 0), gated by scalar gamma, capped at gamma_max
        self.use_skip = arch_cfg.use_skip and self.L >= 2
        self.gamma = arch_cfg.gamma_init
        self.gamma_max = arch_cfg.gamma_max
        if self.use_skip:
            Ws, bs = init_layer(self.sizes_z[0], self.sizes_z[2], arch_cfg.weight_init_scale * 0.5, self.rng)
            self.layers.append(dict(name="skip", W=Ws, b=bs, parent=2, child=0, act_name="identity", is_skip=True))

        # --- lateral quadratic priors Lambda_k, only for the *relaxed* free latents k=1..L-1.
        # (doc2 eq.1/eq.9 defines Lambda for every layer including the leaf and clamped top; here
        #  the leaf (label) is never iteratively relaxed -- see module docstring -- and the clamped
        #  input's own Lambda term is a constant offset that does not affect inference, so neither
        #  needs an explicit Lambda. This is the one place our discriminative re-mapping departs
        #  from doc2's bookkeeping, and is flagged in README "Documented simplifications".)
        self.use_lateral = arch_cfg.use_lateral
        self.lambda_lat = {k: np.full(self.sizes_z[k], arch_cfg.lateral_init) for k in range(1, self.L)} \
            if self.use_lateral else {k: np.zeros(self.sizes_z[k]) for k in range(1, self.L)}

        # --- precisions Pi_k, k = 0..L-1 (isotropic / scalar, doc2 Sec.2 "standing assumptions")
        self.pi = {k: arch_cfg.precision_init for k in range(self.L)}

        self.eps_rel = 1e-2  # layer-relative damping floor for row_scale/col_scale, see layers.py

    # ---------------- generic per-child-index lookups ---------------- #
    def _layers_with_parent(self, p):
        return [l for l in self.layers if l["parent"] == p]

    def _layers_with_child(self, c):
        return [l for l in self.layers if l["child"] == c]

    def act(self, name):
        return ACTIVATIONS[name]

    # ---------------- inference: doc1 INFER_POSTERIORS / doc3 eq.22-23 ---------------- #
    def feedforward_init(self, x):
        """Cheap, sensible starting point for the relaxation: one ordinary forward pass."""
        z = {self.L: x}
        for k in range(self.L - 1, 0, -1):
            ls = self._layers_with_child(k)
            assert len(ls) == 1  # only the main chain ever has child index >= 1
            l = ls[0]
            fn, _ = self.act(l["act_name"])
            pre = z[l["parent"]] @ l["W"].T + l["b"]
            z[k] = fn(pre)
        return z

    def _own_error_and_preact(self, z, k):
        """err_self_k = Pi_k * (z^k - g_k(z^{k+1})); also returns the pre-activation a_k.
        This is the term that drives z^k's OWN relaxation update -- it does NOT get
        multiplied by k's own activation derivative (that factor is only needed when
        this error is being passed *up* to the parent, or used for k's own incoming
        weight gradient -- see `_local_errors` below)."""
        ls = self._layers_with_child(k)
        assert len(ls) == 1
        l = ls[0]
        fn, _ = self.act(l["act_name"])
        pre = z[l["parent"]] @ l["W"].T + l["b"]
        pred = fn(pre)
        err = self.pi[k] * (z[k] - pred)
        return err, pre

    # ---------------- crisp leaf readout: doc1 step 2 / IMAGE_LOSS_LOGIT_GRAD ---------------- #
    def forward_logits(self, z):
        a_y = np.zeros((z[1].shape[0], self.n_classes))
        for l in self._layers_with_child(0):
            pre = z[l["parent"]] @ l["W"].T + l["b"]
            scale = self.gamma if l["is_skip"] else 1.0
            a_y = a_y + scale * pre
        return a_y

    def label_loss_grad(self, a_y, y, T):
        """
        Discriminative analog of doc1's IMAGE_LOSS_LOGIT_GRAD: a crisp dL/dlogit for a
        categorical (softmax) leaf instead of doc1's per-pixel sigmoid/BCE leaf.
        Also plays the role of "err_self_0 (already multiplied by act_0's own
        derivative)" in the biclamped scheme below -- since act_0 is identity, no
        extra factor is needed, and this crisp form is exactly what avoids the
        vanishing-gradient-at-saturation problem doc1 flags for its own leaf.
        Returns (grad_wrt_logits, probs, loss). NOT batch-averaged here -- batch
        averaging happens once, uniformly, in `hebbian_grads`.
        """
        p = softmax(a_y / T)
        y1h = one_hot(y, self.n_classes)
        loss = cross_entropy(p, y)
        grad = (p - y1h) / T
        return grad, p, loss

    # ---------------- standard backprop of the crisp label loss, fixed bug-2 ---------------- #
    def backprop_deltas_from_label(self, z, preacts, y, T):
        """
        Standard backprop of the crisp classification loss from the leaf up through
        the CURRENT z (whether mid-relaxation or fully settled) -- this is doc1's own
        "exactly where PC would backprop" step, now (a) correctly including every
        layer's own activation derivative before the outer product (the missing
        factor was the main correctness bug in the previous version), and (b) called
        fresh at EVERY relaxation iteration when `y` is known, not just once at the
        end -- which is what makes the relaxation genuinely biclamped: the label's
        influence reaches the hidden latents WHILE they're settling, exactly like a
        clamped leaf in classical supervised predictive coding (Whittington & Bogacz
        2017), not just in a one-shot pass after the fact.

        Returns delta_z (dLoss/dz^k) and delta_pre (dLoss/dpre_k = delta_z_k * der_k)
        for every child index k=0..L-1. `delta_pre` is exactly what `hebbian_grads`
        needs, and what gets ADDED into the relaxation's own gradient on z^k (see
        `infer` below) -- both are plain backprop quantities with an unambiguous sign
        convention (no separate free-energy sign convention to keep consistent with).
        """
        a_y = self.forward_logits(z)
        g_y, p, loss = self.label_loss_grad(a_y, y, T)
        delta_z = {0: g_y}
        delta_pre = {0: g_y}  # act_0 is identity, so dLoss/dz^0 == dLoss/dpre_0
        for parent in range(1, self.L):
            total = 0.0
            for l in self._layers_with_parent(parent):
                scale = self.gamma if l["is_skip"] else 1.0
                total = total + scale * (delta_pre[l["child"]] @ l["W"])
            delta_z[parent] = total
            _, der_fn = self.act(self._layers_with_child(parent)[0]["act_name"])
            delta_pre[parent] = delta_z[parent] * der_fn(preacts[parent])
        return delta_z, delta_pre, p, loss

    # ---------------- inference: doc1 INFER_POSTERIORS / doc3 eq.22-23, biclamped ---------------- #
    def infer(self, x, y=None, T=1.0, infer_cfg=None):
        """
        Synchronous (Jacobi-style) fixed-point relaxation of the free latents
        z^1..z^{L-1}, given the clamped input z^L = x. Each iteration's gradient on
        z^k is the usual top-down PC free-energy term (own prediction error + lateral
        prior - bottom-up error from the hidden layer below), UNCHANGED from the
        original design.

        If `y` is given (TRAIN time) AND `infer_cfg.biclamp_during_relaxation` is
        True, the standard-backprop gradient of the actual classification loss w.r.t.
        z^k is ALSO injected into every iteration -- genuine biclamped/bidirectional
        message passing, computed fresh from the CURRENT (possibly unsettled) z. This
        is OFF by default: we benchmarked it and found it drives training loss down
        very fast while making validation accuracy WORSE (letting inference "see" the
        label lets the hidden layer shortcut to a low loss for that batch without the
        weights actually improving, which doesn't transfer to test time). See
        InferenceConfig.biclamp_during_relaxation and the README for the numbers.

        Regardless of that flag, if `y` is given the label IS always used once, in a
        single backward pass AFTER the hidden latents settle, to compute the weight
        gradients (see the final lines of this method / `hebbian_grads`) -- this part
        is never skipped, since it's how the network learns anything at all.

        At TEST time (`y=None`) only x's top-down influence drives the relaxation,
        and the prediction is read off once via `forward_logits` after settling --
        the standard, unavoidable train/test asymmetry of every biclamped/supervised
        PC network in the literature (Whittington & Bogacz 2017; doc3's "standard
        PC", eq.22-24).

        Returns (z, preacts, delta_pre) where `delta_pre` (None at test time) is
        exactly what `hebbian_grads` needs -- computed once more at the final
        settled z, no separate backward pass required beyond this.
        """
        cfg = infer_cfg or self.infer_cfg
        z = self.feedforward_init(x)
        z_prev = {k: z[k].copy() for k in range(1, self.L)}
        inject_during_relaxation = y is not None and cfg.biclamp_during_relaxation

        for _ in range(cfg.k_infer):
            err_self, preacts = {}, {}
            for k in range(1, self.L):
                err_self[k], preacts[k] = self._own_error_and_preact(z, k)

            delta_z_label = {}
            if inject_during_relaxation:
                delta_z_label, _, _, _ = self.backprop_deltas_from_label(z, preacts, y, T)

            new_z = {}
            for k in range(1, self.L):
                lateral = self.lambda_lat[k] * z[k]
                if k - 1 >= 1:
                    l_child = self._layers_with_child(k - 1)[0]
                    _, der_fn = self.act(l_child["act_name"])
                    bottom_up_pc = (der_fn(preacts[k - 1]) * err_self[k - 1]) @ l_child["W"]
                else:
                    bottom_up_pc = 0.0
                label_term = delta_z_label.get(k, 0.0)
                dF_dz = err_self[k] + lateral - bottom_up_pc + label_term
                precond = 1.0 / (self.pi[k] + self.lambda_lat[k] + 1.0)
                step = -cfg.step_size * precond * dF_dz
                momentum = cfg.mu_polyak * (z[k] - z_prev[k])
                l1 = -cfg.lambda_l1_z * np.sign(z[k])
                new_z[k] = z[k] + step + momentum + l1
                new_z[k] = np.clip(new_z[k], -cfg.z_clip, cfg.z_clip)
            z_prev = {k: z[k] for k in range(1, self.L)}
            for k in range(1, self.L):
                z[k] = new_z[k]

        preacts = {}
        for k in range(1, self.L):
            _, preacts[k] = self._own_error_and_preact(z, k)
        delta_pre = None
        if y is not None:
            _, delta_pre, _, _ = self.backprop_deltas_from_label(z, preacts, y, T)
        return z, preacts, delta_pre

    def hebbian_grads(self, z, delta_pre):
        """dW_l = mean_batch( outer(delta_pre[child(l)], z[parent(l)]) ) -- standard
        backprop weight gradient (doc1 step 3.1 / doc3 eq.24), now correctly including
        each child's own activation derivative (the previous bug)."""
        grads = {}
        n = z[self.L].shape[0]
        for l in self.layers:
            if l["child"] not in delta_pre:
                continue
            pre = z[l["parent"]]
            d = delta_pre[l["child"]]
            scale = self.gamma if l["is_skip"] else 1.0
            gW = scale * (d.T @ pre) / n
            gb = scale * d.sum(axis=0) / n
            grads[l["name"]] = (gW, gb)
        return grads

    # ---------------- shared apply step (regularize, clip, descend) ---------------- #
    def _apply_update(self, name, dW, db, optim_cfg):
        l = next(x for x in self.layers if x["name"] == name)
        dW = dW - optim_cfg.lambda_l2 * l["W"] - optim_cfg.lambda_l1 * np.sign(l["W"])
        dW = np.clip(dW, -optim_cfg.grad_clip, optim_cfg.grad_clip)
        db = np.clip(db, -optim_cfg.grad_clip, optim_cfg.grad_clip)
        l["W"] = l["W"] - optim_cfg.eta * dW
        l["b"] = l["b"] - optim_cfg.eta * db

    def get_weight_matrices(self):
        return [l["W"] for l in self.layers if not l["is_skip"]]

    def get_hidden_activations(self, X, layer_idx=1, infer_cfg=None):
        """Always uses y=None (top-down only) -- representation/entanglement metrics
        must reflect what the network does WITHOUT seeing the answer, exactly as at
        real test time, or they'd trivially leak label information."""
        z, _, _ = self.infer(X, y=None, infer_cfg=infer_cfg)
        return z[layer_idx]

    def predict(self, X, T=1.0, infer_cfg=None):
        z, _, _ = self.infer(X, y=None, T=T, infer_cfg=infer_cfg)
        a_y = self.forward_logits(z)
        return softmax(a_y / T)

    def compute_grads_no_update(self, X, y, T=1.0):
        """Runs inference (x clamped at the top; the label is used, by default, only
        in a single backward pass after the hidden latents settle -- see
        InferenceConfig.biclamp_during_relaxation for the opt-in alternative) and
        returns the standard-backprop weight gradients, WITHOUT applying any update.
        Used both by train_step (which then applies the update) and by the
        influence-consistency metric, which needs the same gradients computed
        repeatedly with frozen weights."""
        z, preacts, delta_pre = self.infer(X, y=y, T=T)
        a_y = self.forward_logits(z)
        p = softmax(a_y / T)
        loss = cross_entropy(p, y)
        grads = self.hebbian_grads(z, delta_pre)
        return dict(z=z, preacts=preacts, delta_pre=delta_pre, a_y=a_y, p=p, loss=loss, grads=grads)


# --------------------------------------------------------------------------- #
# Vanilla discriminative PC: Hebbian update, no causal gate, no clarity
# (doc3 Sec 2.4/3.5: "standard PC uses only first-order quantities")
# --------------------------------------------------------------------------- #
class PCNetwork(BaseHierarchical):
    def __init__(self, arch_cfg, infer_cfg, n_classes, input_dim, optim_cfg, seed=0):
        super().__init__(arch_cfg, infer_cfg, n_classes, input_dim, seed)
        self.optim = optim_cfg
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def train_step(self, x, y, T=1.0):
        out = self.compute_grads_no_update(x, y, T)
        for l in self.layers:
            gW, gb = out["grads"][l["name"]]
            self._apply_update(l["name"], gW, gb, self.optim)

        if self.use_skip:
            l = next(x_ for x_ in self.layers if x_["is_skip"])
            raw_pre = out["z"][2] @ l["W"].T + l["b"]
            d_gamma = float(np.mean(np.sum(out["delta_pre"][0] * raw_pre, axis=1)))
            self.gamma = float(np.clip(self.gamma - self.optim.eta * d_gamma, 0.0, self.gamma_max))

        return out["loss"]


# --------------------------------------------------------------------------- #
# Causal Coding: PC + naturalized influence M_l + amplitude-preserving gate
# G_l + diffusion-based clarity D_l (doc1 steps 3.2-3.4), driven by SCHEDULE.
# --------------------------------------------------------------------------- #
class CCNetwork(BaseHierarchical):
    def __init__(self, arch_cfg, infer_cfg, n_classes, input_dim, optim_cfg, seed=0):
        super().__init__(arch_cfg, infer_cfg, n_classes, input_dim, seed)
        self.optim = optim_cfg
        self.history = {
            "train_loss": [], "val_loss": [], "val_acc": [],
            "gate_mean": [], "gate_std": [], "gate_suppressed_frac": [],
            "alpha_t": [], "p_t": [], "lambda_diff_t": [], "clarity_on": [],
        }
        self._batch_count = 0
        self.last_M = {}   # most recent naturalized influence per layer (for influence-consistency metric)
        self.last_G = {}   # most recent gate per layer

    # -------- doc1 step 3.2: naturalized influence M_l -------- #
    def _child_derivative(self, child_idx, a_y=None, p=None, T=1.0, z=None, preacts=None):
        """ACTIVATION_DERIVATIVE for whichever activation sits at `child_idx`."""
        if child_idx == 0:
            # leaf likelihood model is softmax+CE at temperature T -- the curvature-relevant
            # "activation derivative" is the softmax Jacobian's diagonal, p*(1-p)/T, the direct
            # categorical analogue of doc1's "sigmoid: der = sig(a)*(1-sig(a))" choice.
            return (p * (1.0 - p)) / T
        l = self._layers_with_child(child_idx)[0]
        _, der_fn = self.act(l["act_name"])
        return der_fn(preacts[child_idx])

    def _naturalized_influence(self, layer, z, p, T, preacts):
        der = self._child_derivative(layer["child"], p=p, T=T, preacts=preacts)
        row_scale = row_scale_from_child_derivatives(der, eps_rel=self.eps_rel)      # (n_child,)
        pre = z[layer["parent"]]
        col_scale = col_scale_from_parent(pre, eps_rel=self.eps_rel)                  # (n_parent,)
        M = row_scale[:, None] * np.abs(layer["W"]) * col_scale[None, :]
        return M

    def _gate(self, M, n_in, alpha_t, p_t, floor_frac):
        R = normalize_inputwise(np.abs(M) ** p_t)
        G = 1.0 + alpha_t * (n_in * R - 1.0)
        G = np.maximum(G, 1.0 - alpha_t + floor_frac * alpha_t)
        return stop_gradient(G)

    # -------- doc1 step 3.4: clarity from a diffused influence graph -------- #
    def _build_block_adjacency(self, M_by_layer):
        offsets = {}
        cursor = 0
        for k in range(self.L + 1):
            offsets[k] = cursor
            cursor += self.sizes_z[k]
        total = cursor
        A = np.zeros((total, total))
        for name, M in M_by_layer.items():
            l = next(x for x in self.layers if x["name"] == name)
            P = normalize_inputwise(M)  # plain normalize, NOT raised to p_t (doc1 step3.4 uses M_l directly)
            c0, p0 = offsets[l["child"]], offsets[l["parent"]]
            A[c0:c0 + P.shape[0], p0:p0 + P.shape[1]] += P
        return A, offsets

    def _clarity_terms(self, M_by_layer, order, delta=1e-3):
        A, offsets = self._build_block_adjacency(M_by_layer)
        K = np.zeros_like(A)
        Apow = A.copy()
        for i in range(order):
            K = K + Apow
            if i < order - 1:
                Apow = Apow @ A
        D = {}
        for name, M in M_by_layer.items():
            l = next(x for x in self.layers if x["name"] == name)
            P = normalize_inputwise(M)
            c0, p0 = offsets[l["child"]], offsets[l["parent"]]
            direct = A[c0:c0 + P.shape[0], p0:p0 + P.shape[1]]
            diffused = K[c0:c0 + P.shape[0], p0:p0 + P.shape[1]]
            os_l = np.maximum(diffused - direct - delta, 0.0)
            D[name] = os_l * P * np.sign(l["W"])
        return D

    def compute_M_no_update(self, x, y, T=1.0):
        """Convenience wrapper used by the influence-consistency metric: returns both the
        raw (pre-gate) Hebbian gradients AND the naturalized influence M_l, for the same batch."""
        out = self.compute_grads_no_update(x, y, T)
        M_by_layer = {}
        for l in self.layers:
            M_by_layer[l["name"]] = self._naturalized_influence(l, out["z"], out["p"], T, out["preacts"])
        out["M"] = M_by_layer
        return out

    def train_step(self, x, y, sched_state: dict):
        out = self.compute_grads_no_update(x, y, sched_state["T_t"])
        z, preacts, p, grads = out["z"], out["preacts"], out["p"], out["grads"]

        M_by_layer = {}
        for l in self.layers:
            M_by_layer[l["name"]] = self._naturalized_influence(l, z, p, sched_state["T_t"], preacts)
        self.last_M = M_by_layer

        gate_vals = []
        gated_grads = {}
        for l in self.layers:
            M = M_by_layer[l["name"]]
            n_in = l["W"].shape[1]
            G = self._gate(M, n_in, sched_state["alpha_t"], sched_state["p_t"], sched_state["gate_floor_min_fraction"])
            self.last_G[l["name"]] = G
            gW, gb = grads[l["name"]]
            # Gated pruning: the gate above only freezes adaptation for suppressed
            # connections (they stay near their nonzero random init); this term adds
            # genuine shrinkage proportional to (1 - G), so connections CC has
            # persistently flagged as low-influence actually decay toward zero instead
            # of just sitting still. 0 for any connection with G>=1 (boosted/neutral).
            prune = self.optim.lambda_l1_gated * np.maximum(1.0 - G, 0.0) * np.sign(l["W"])
            gated_grads[l["name"]] = (G * gW + prune, gb)
            gate_vals.append(G)

        if sched_state["clarity_on"] and (self._batch_count % sched_state["clarity_every_n_batches"] == 0):
            D_by_layer = self._clarity_terms(M_by_layer, sched_state["diffusion_order"])
            for l in self.layers:
                gW, gb = gated_grads[l["name"]]
                gated_grads[l["name"]] = (gW + sched_state["lambda_diff_t"] * D_by_layer[l["name"]], gb)

        for l in self.layers:
            gW, gb = gated_grads[l["name"]]
            self._apply_update(l["name"], gW, gb, self.optim)

        if self.use_skip:
            l = next(x_ for x_ in self.layers if x_["is_skip"])
            raw_pre = z[2] @ l["W"].T + l["b"]
            d_gamma = float(np.mean(np.sum(out["delta_pre"][0] * raw_pre, axis=1)))
            self.gamma = float(np.clip(self.gamma - self.optim.eta * d_gamma, 0.0, self.gamma_max))

        self._batch_count += 1
        all_g = np.concatenate([g.ravel() for g in gate_vals])
        return out["loss"], dict(gate_mean=float(all_g.mean()), gate_std=float(all_g.std()),
                                  gate_suppressed_frac=float(np.mean(all_g < 1.0 - 0.5 * sched_state["alpha_t"])))
