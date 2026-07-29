"""Numerical gradient check for nettalk/prototype.py's prototype cross-entropy loss.

Verifies:
1. The analytic dL/do for prototype_ce_loss_and_grad against central finite
   differences, for both the phoneme (21-dim) and stress (5-dim) blocks.
2. An end-to-end finite-difference check of the full per-word loss against the
   backprop gradients produced by _compute_word_gradients_prototype (through the
   sigmoid output layer and a single hidden layer), on a tiny random network.

Run: python scripts/check_prototype_gradient.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nettalk.data import load_feature_mapping
from nettalk.model import NettalkMLP
from nettalk.prototype import (
    PrototypeTrainingConfig,
    _compute_word_gradients_prototype,
    build_prototype_tables,
    prototype_ce_loss_and_grad,
)

FD_EPS = 1e-6
TOLERANCE = 1e-6
RNG = np.random.default_rng(0)


def _relative_error(analytic: np.ndarray, numeric: np.ndarray) -> float:
    denom = np.maximum(np.abs(analytic), np.abs(numeric))
    denom = np.where(denom < 1e-8, 1.0, denom)
    return float(np.max(np.abs(analytic - numeric) / denom))


def check_block_gradient(name: str, dim: int, num_classes: int, trials: int = 20) -> bool:
    prototypes = RNG.integers(0, 2, size=(num_classes, dim)).astype(np.float64)
    zero_rows = np.all(prototypes == 0, axis=1)
    prototypes[zero_rows, 0] = 1.0
    prototypes_norm = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)

    worst = 0.0
    for _ in range(trials):
        o = RNG.uniform(0.05, 0.95, size=(dim,))
        target = int(RNG.integers(0, num_classes))
        _, analytic_grad = prototype_ce_loss_and_grad(o, prototypes_norm, target, tau=0.1)

        numeric_grad = np.zeros_like(o)
        for i in range(dim):
            plus = o.copy()
            plus[i] += FD_EPS
            minus = o.copy()
            minus[i] -= FD_EPS
            loss_plus, _ = prototype_ce_loss_and_grad(plus, prototypes_norm, target, tau=0.1)
            loss_minus, _ = prototype_ce_loss_and_grad(minus, prototypes_norm, target, tau=0.1)
            numeric_grad[i] = (loss_plus - loss_minus) / (2 * FD_EPS)

        worst = max(worst, _relative_error(analytic_grad, numeric_grad))

    passed = worst < TOLERANCE
    print(f"[{name} block, dim={dim}, classes={num_classes}] max rel error over {trials} trials = {worst:.3e} -> {'PASS' if passed else 'FAIL'}")
    return passed


def check_end_to_end_backprop(mse_weight: float = 0.0) -> bool:
    mapping = load_feature_mapping(".")
    tables = build_prototype_tables(mapping)
    config = PrototypeTrainingConfig(tau=0.15, stress_weight=1.0, mse_weight=mse_weight)

    input_dim = 10
    hidden_dim = 4
    model = NettalkMLP(input_dim=input_dim, hidden_dim=hidden_dim, seed=123)

    word_len = 3
    x_word = RNG.uniform(-1.0, 1.0, size=(word_len, input_dim))
    phoneme_indices = RNG.integers(0, len(tables.phoneme_symbols), size=word_len)
    stress_indices = RNG.integers(0, len(tables.stress_symbols), size=word_len)
    phoneme_word = "".join(tables.phoneme_symbols[int(i)] for i in phoneme_indices)
    stress_word = "".join(tables.stress_symbols[int(i)] for i in stress_indices)
    # Full 26-dim target codes for the sampled phoneme+stress, matching how the
    # dataset stores per-letter targets (mapping.vector_for).
    y_word = np.stack([mapping.vector_for(phoneme_word[i], stress_word[i]) for i in range(word_len)], axis=0)

    def sum_loss_for_model(candidate_model: NettalkMLP) -> float:
        _, _, avg_loss = _compute_word_gradients_prototype(candidate_model, x_word, phoneme_word, stress_word, tables, config, y_word)
        return avg_loss * word_len

    grad_weights, grad_biases, _ = _compute_word_gradients_prototype(model, x_word, phoneme_word, stress_word, tables, config, y_word)
    analytic = {"W1": grad_weights[0], "W2": grad_weights[1], "b1": grad_biases[0], "b2": grad_biases[1]}

    checks = [
        ("W1", model.W1, (0, 0)),
        ("W1", model.W1, (3, 2)),
        ("W1", model.W1, (9, 3)),
        ("W2", model.W2, (1, 5)),
        ("W2", model.W2, (2, 10)),
        ("W2", model.W2, (3, 25)),
        ("b1", model.b1, (2,)),
        ("b2", model.b2, (7,)),
    ]

    worst = 0.0
    for name, param, idx in checks:
        original = param[idx]
        param[idx] = original + FD_EPS
        loss_plus = sum_loss_for_model(model)
        param[idx] = original - FD_EPS
        loss_minus = sum_loss_for_model(model)
        param[idx] = original

        numeric_grad = (loss_plus - loss_minus) / (2 * FD_EPS)
        analytic_grad = float(analytic[name][idx])
        rel_error = abs(analytic_grad - numeric_grad) / max(abs(analytic_grad), abs(numeric_grad), 1e-8)
        worst = max(worst, rel_error)
        print(f"  {name}{idx}: analytic={analytic_grad:.6e} numeric={numeric_grad:.6e} rel_err={rel_error:.3e}")

    passed = worst < TOLERANCE
    tag = f"mse_weight={mse_weight}"
    print(f"[end-to-end backprop, input_dim={input_dim}, hidden={hidden_dim}, {tag}] max rel error = {worst:.3e} -> {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> None:
    print("Running prototype cross-entropy gradient checks...")
    results = [
        check_block_gradient("phoneme", dim=21, num_classes=55),
        check_block_gradient("stress", dim=5, num_classes=6),
        check_end_to_end_backprop(mse_weight=0.0),
        check_end_to_end_backprop(mse_weight=1.0),
    ]
    overall = all(results)
    print()
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    if not overall:
        sys.exit(1)


if __name__ == "__main__":
    main()
