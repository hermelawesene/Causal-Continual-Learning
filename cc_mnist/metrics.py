"""
metrics.py
==========
Every quantitative metric used to compare MLP vs PC vs CC. Organized into
the five families the user asked for, plus a couple of extras noted
inline. Every function takes plain numpy arrays / model objects and
returns plain Python numbers, lists, or small numpy arrays -- nothing
here is tied to a particular model class, so the same functions are
called identically for MLP, PC and CC in run_experiment.py.
"""
from collections import defaultdict
import numpy as np
from sklearn.metrics import confusion_matrix, mutual_info_score
from sklearn.manifold import TSNE

from utils import accuracy, cross_entropy


# =========================================================================== #
# (1) Standard performance
# =========================================================================== #
def performance_metrics(probs: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    pred = np.argmax(probs, axis=1)
    acc = accuracy(probs, y)
    loss = cross_entropy(probs, y)
    cm = confusion_matrix(y, pred, labels=list(range(n_classes)))
    per_class_acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    ece = expected_calibration_error(probs, y)
    return dict(accuracy=acc, loss=loss, confusion_matrix=cm, per_class_accuracy=per_class_acc, ece=ece)


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Standard ECE: |confidence - accuracy| averaged over confidence bins, weighted by bin mass."""
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


# =========================================================================== #
# (2) Sparsity / modularity
# =========================================================================== #
def weight_sparsity(W: np.ndarray, rel_threshold: float = 0.02) -> float:
    """Fraction of weights with |w| below `rel_threshold` * the layer's own max |w|."""
    scale = np.max(np.abs(W)) + 1e-12
    return float(np.mean(np.abs(W) < rel_threshold * scale))


def effective_connectivity(W: np.ndarray, rel_threshold: float = 0.02) -> float:
    """Average number of 'active' (non-near-zero) input connections per output neuron."""
    scale = np.max(np.abs(W)) + 1e-12
    active = np.abs(W) >= rel_threshold * scale
    return float(active.sum(axis=1).mean())


def effective_rank(W: np.ndarray) -> float:
    """exp(entropy of normalized singular values) -- a continuous, scale-free 'how many directions
    does this weight matrix actually use' measure. Low effective rank => more modular/structured."""
    s = np.linalg.svd(W, compute_uv=False)
    s = s[s > 1e-12]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def sparsity_report(weight_matrices: list, rel_threshold: float = 0.02) -> dict:
    per_layer = []
    for i, W in enumerate(weight_matrices):
        per_layer.append(dict(
            layer=i, shape=W.shape,
            sparsity=weight_sparsity(W, rel_threshold),
            effective_connectivity=effective_connectivity(W, rel_threshold),
            effective_rank=effective_rank(W),
            max_possible_connectivity=W.shape[1],
        ))
    overall_sparsity = float(np.mean([p["sparsity"] for p in per_layer]))
    return dict(per_layer=per_layer, overall_sparsity=overall_sparsity)


# =========================================================================== #
# (3) Entanglement
# =========================================================================== #
def class_conditional_activation_profile(activations: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """Returns (n_classes, n_neurons): mean activation of each neuron, per class."""
    profile = np.zeros((n_classes, activations.shape[1]))
    for c in range(n_classes):
        mask = y == c
        if mask.sum() > 0:
            profile[c] = activations[mask].mean(axis=0)
    return profile


def neuron_entanglement_scores(activations: np.ndarray, y: np.ndarray, n_classes: int,
                                 frac_threshold: float = 0.5) -> dict:
    """
    For each neuron, count how many classes drive it to within `frac_threshold` of its own
    peak class-mean response. A neuron that is "selective" responds strongly to ~1 class
    (low entanglement); a neuron that is "entangled" responds broadly to many unrelated
    classes (high entanglement). We use the *absolute* class-mean profile so that both
    strongly-positive and strongly-negative (tanh) responses count as "responding".
    """
    profile = np.abs(class_conditional_activation_profile(activations, y, n_classes))
    peak = profile.max(axis=0) + 1e-12
    responds = profile >= frac_threshold * peak[None, :]
    n_responding_classes = responds.sum(axis=0)  # per neuron
    return dict(
        per_neuron_n_classes=n_responding_classes,
        mean_entanglement=float(n_responding_classes.mean()),
        frac_highly_entangled=float(np.mean(n_responding_classes >= int(0.5 * n_classes))),
    )


def neuron_label_mutual_information(activations: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    """Discretized mutual information between each neuron's activation and the true label."""
    n_neurons = activations.shape[1]
    mis = np.zeros(n_neurons)
    for j in range(n_neurons):
        col = activations[:, j]
        edges = np.linspace(col.min() - 1e-9, col.max() + 1e-9, n_bins + 1)
        binned = np.digitize(col, edges) - 1
        mis[j] = mutual_info_score(y, binned)
    return dict(per_neuron_mi=mis, mean_mi=float(mis.mean()))


def pairwise_activation_correlation(activations: np.ndarray) -> dict:
    """Mean |off-diagonal correlation| between neurons -- a coarse 'redundancy / co-activation' proxy."""
    if activations.shape[1] < 2:
        return dict(corr_matrix=np.array([[1.0]]), mean_abs_offdiag=0.0)
    std = activations.std(axis=0)
    keep = std > 1e-8
    A = activations[:, keep]
    corr = np.corrcoef(A, rowvar=False)
    n = corr.shape[0]
    off = corr[~np.eye(n, dtype=bool)]
    return dict(corr_matrix=corr, mean_abs_offdiag=float(np.mean(np.abs(off))))


def entanglement_report(activations: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    ent = neuron_entanglement_scores(activations, y, n_classes)
    mi = neuron_label_mutual_information(activations, y)
    corr = pairwise_activation_correlation(activations)
    return dict(
        mean_entanglement=ent["mean_entanglement"],
        frac_highly_entangled=ent["frac_highly_entangled"],
        per_neuron_n_classes=ent["per_neuron_n_classes"],
        mean_neuron_label_mi=mi["mean_mi"],
        per_neuron_mi=mi["per_neuron_mi"],
        mean_abs_pairwise_corr=corr["mean_abs_offdiag"],
        corr_matrix=corr["corr_matrix"],
    )


# =========================================================================== #
# (4) Influence consistency (CC-specific: stability of "what matters" across batches)
# =========================================================================== #
def flat_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def batch_to_batch_consistency(importance_sequence: list) -> dict:
    """
    `importance_sequence` is a list of same-shaped arrays, one per consecutive
    minibatch (e.g. |Hebbian dW| for the plain-gradient control, or M_l / G_l
    for CC). Returns the mean cosine similarity between consecutive batches'
    importance maps -- a higher value means the model assigns importance to
    the same connections batch after batch (more *stable* credit assignment).
    """
    sims = [flat_cosine(importance_sequence[i], importance_sequence[i + 1])
            for i in range(len(importance_sequence) - 1)]
    return dict(mean_consistency=float(np.mean(sims)) if sims else float("nan"),
                std_consistency=float(np.std(sims)) if sims else float("nan"),
                sims=sims)


def influence_consistency_report(grad_sequence: dict, cc_influence_sequence: dict = None) -> dict:
    """
    `grad_sequence` / `cc_influence_sequence`: dict layer_name -> list of arrays across batches.
    Returns, per layer, the batch-to-batch consistency of the raw Hebbian-gradient magnitude
    (the backprop-style control) vs the CC naturalized-influence / gate (the treatment).
    """
    out = {"raw_gradient": {}, "cc_influence": {}}
    for name, seq in grad_sequence.items():
        out["raw_gradient"][name] = batch_to_batch_consistency([np.abs(g) for g in seq])
    if cc_influence_sequence is not None:
        for name, seq in cc_influence_sequence.items():
            out["cc_influence"][name] = batch_to_batch_consistency([np.abs(g) for g in seq])
    return out


# =========================================================================== #
# (5) Representation structure (t-SNE of hidden activations)
# =========================================================================== #
def tsne_embedding(activations: np.ndarray, n_samples: int = 1500, seed: int = 0) -> np.ndarray:
    n = activations.shape[0]
    if n > n_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, n_samples, replace=False)
        activations = activations[idx]
    perplexity = min(30, max(5, activations.shape[0] // 100))
    emb = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(activations)
    return emb


def silhouette_of_classes(embedding: np.ndarray, y: np.ndarray) -> float:
    """Silhouette score of the 2D embedding w.r.t. true class labels -- a single number
    summarizing 'how cleanly clustered by class does this representation look'."""
    from sklearn.metrics import silhouette_score
    try:
        return float(silhouette_score(embedding, y))
    except Exception:
        return float("nan")
