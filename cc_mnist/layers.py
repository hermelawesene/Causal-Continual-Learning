"""
layers.py
=========
Elementary numpy building blocks. These intentionally mirror, almost
line-for-line, the small subroutines given in
"Causal Coding (CC) Training Loop: Fully Commented Pseudocode" (doc1):

    ROW_SCALE_FROM_CHILD_DERIVATIVES   -> row_scale_from_child_derivatives
    NORMALIZE_INPUTWISE                -> normalize_inputwise
    (weight init, activations)         -> ACTIVATIONS, init_layer

Keeping these as free functions (rather than burying them inside a model
class) makes the doc1 <-> code correspondence easy to audit.
"""
import numpy as np


def _tanh(a):
    return np.tanh(a)


def _tanh_der(a):
    t = np.tanh(a)
    return 1.0 - t * t


def _identity(a):
    return a


def _identity_der(a):
    return np.ones_like(a)


# name -> (activation fn, derivative fn). "identity" is used for the
# output/logit layer (a_0 in our notation): its prediction IS the logit,
# matching doc1's "image logits before sigmoid" treatment of a_img.
ACTIVATIONS = {
    "tanh": (_tanh, _tanh_der),
    "identity": (_identity, _identity_der),
}


def init_layer(n_out: int, n_in: int, scale: float, rng: np.random.RandomState):
    """Xavier-ish init. W has shape (n_out, n_in) so that W @ z_parent -> child pre-activation."""
    limit = scale * np.sqrt(6.0 / (n_in + n_out))
    W = rng.uniform(-limit, limit, size=(n_out, n_in))
    b = np.zeros(n_out)
    return W, b


def row_scale_from_child_derivatives(der: np.ndarray, eps_rel: float = 1e-2) -> np.ndarray:
    """
    doc1 ROW_SCALE_FROM_CHILD_DERIVATIVES:
        s1 = MEAN(ABS(der), axis=batch_and_spatial)
        s2 = MEAN(der^2,   axis=batch_and_spatial) + eps
        RETURN s1 / s2

    `der` has shape (batch, n_out) -- the elementwise activation derivative
    of the CHILD layer's own activation, evaluated at its own pre-activation.
    Returns a (n_out,) vector, one scale per output ("row" of W_l).

    IMPORTANT damping note: doc1's "+eps" is a placeholder for what in
    practice must be a *layer-relative* damping term, not a tiny absolute
    constant. A saturated tanh unit has der^2 -> 0, and an absolute eps
    (e.g. 1e-6) would then blow s1/s2 up to ~1e6 for that single output --
    i.e. the *most dead* units would get reported as the *most influential*,
    exactly backwards from the intent. This is the same damping issue K-FAC
    and other natural-gradient approximations face with near-zero curvature
    directions (Martens & Grosse), and the standard fix is the one used
    here: scale the floor to the layer's own typical curvature instead of
    an absolute constant.
    """
    s1 = np.mean(np.abs(der), axis=0)
    s2 = np.mean(der ** 2, axis=0)
    floor = eps_rel * np.mean(s2) + 1e-12
    return s1 / np.maximum(s2, floor)


def col_scale_from_parent(pre: np.ndarray, eps_rel: float = 1e-2) -> np.ndarray:
    """doc1 step 3.2: col_scale = 1 / sqrt(E[pre^2] + eps). `pre` shape (batch, n_in) -> (n_in,).
    Same layer-relative damping as `row_scale_from_child_derivatives` above, and for the
    same reason: an always-near-zero input (e.g. an MNIST border pixel that's background
    in every image) would otherwise get an enormous col_scale, making CC treat dead
    pixels as highly "influential" -- the opposite of what the gate is supposed to do."""
    var = np.mean(pre ** 2, axis=0)
    floor = eps_rel * np.mean(var) + 1e-12
    return 1.0 / np.sqrt(np.maximum(var, floor))


def normalize_inputwise(T: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    doc1 NORMALIZE_INPUTWISE: normalize per output row, summing over inputs.
    T has shape (n_out, n_in) (matches a weight-matrix-shaped tensor, e.g. M_l or |M_l|^p).
    """
    denom = np.sum(T, axis=1, keepdims=True) + eps
    return T / denom


def stop_gradient(x: np.ndarray) -> np.ndarray:
    """
    doc1 step 3.3: "Do NOT backprop through gates". We have no autodiff graph
    in this pure-numpy implementation, so this is a documentation no-op --
    it exists purely so call sites read the same way the pseudocode does,
    and so that a future autodiff port (e.g. to torch) knows exactly where
    a `.detach()` / `stop_gradient` call is required.
    """
    return x
