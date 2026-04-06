"""
Neural hyper-heuristic (chapter 3).

Architecture: 8 → 6 → 4 → 2 → 1  (ReLU hidden, Sigmoid output)
Total trainable parameters: 95  (matches chromosome length in GA config)

Input features (8):  subset of the 13 metrics — indices 0..7
  0 WOD, 1 WOD_2, 2 rank, 3 C, 4 |pred|, 5 |succ|, 6 TW_in, 7 TW_out

Output: scalar priority in (0, 1) — higher means schedule sooner.
"""

from __future__ import annotations
import numpy as np

# feature indices within the 13-metric vector used by the NN
NN_FEATURE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7]   # WOD..TW_out
NN_INPUT_DIM = len(NN_FEATURE_INDICES)            # 8

# layer widths: input(8) → 6 → 4 → 2 → 1
_LAYERS = [NN_INPUT_DIM, 6, 4, 2, 1]

# total params: sum of (in+1)*out for each consecutive pair
CHROMOSOME_LEN: int = sum(
    (_LAYERS[i] + 1) * _LAYERS[i + 1]
    for i in range(len(_LAYERS) - 1)
)  # = 54 + 28 + 10 + 3 = 95


def _unpack(weights: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split flat chromosome into (W, b) pairs for each layer."""
    params: list[tuple[np.ndarray, np.ndarray]] = []
    offset = 0
    for i in range(len(_LAYERS) - 1):
        in_dim, out_dim = _LAYERS[i], _LAYERS[i + 1]
        w_size = in_dim * out_dim
        W = weights[offset: offset + w_size].reshape(in_dim, out_dim)
        offset += w_size
        b = weights[offset: offset + out_dim]
        offset += out_dim
        params.append((W, b))
    return params


def forward(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Forward pass for a batch of feature vectors.

    x:       (n, 8) float array — one row per ready task
    weights: (95,) chromosome vector
    returns: (n,) priority scores in (0, 1)
    """
    params = _unpack(weights)
    h = x
    for i, (W, b) in enumerate(params):
        h = h @ W + b
        if i < len(params) - 1:
            h = np.maximum(h, 0.0)   # ReLU
        else:
            h = 1.0 / (1.0 + np.exp(-h))  # Sigmoid
    return h.reshape(-1)


def score_tasks(task_indices: list[int], metrics: np.ndarray,
                weights: np.ndarray) -> np.ndarray:
    """
    Score a list of ready tasks.

    task_indices: rustworkx node indices
    metrics:      full (max_idx+1, 13) normalised metric array
    weights:      (95,) chromosome

    Returns (n,) priority scores.
    """
    feats = metrics[task_indices][:, NN_FEATURE_INDICES]
    return forward(feats, weights)
