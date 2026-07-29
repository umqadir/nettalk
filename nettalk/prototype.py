"""Prototype cross-entropy training for the NETtalk single-hidden-layer MLP.

Standard NETtalk training uses per-bit MSE through the sigmoid output layer. This
module trains the SAME sigmoid output representation (a distributed 26-dim feature
vector in [0,1]^26, exactly like a 1987 phoneme code) but replaces the loss with a
cosine-similarity softmax cross-entropy that directly optimizes what the paper's
"best guess" decoder actually measures: nearest-angle match to a phoneme/stress
prototype code (see nettalk/eval.py: _best_guess_accuracy).

For an output block o (post-sigmoid, o in [0,1]^D) and unit-normalized prototype
codes p_hat_k:
    o_hat = o / ||o||
    logit_k = (o_hat . p_hat_k) / tau
    q = softmax_k(logit_k)
    L = -log q_target

The phoneme block (21 dims) and stress block (5 dims) are scored independently
against their own prototype inventories and summed (stress weighted by
stress_weight). Training mechanics (per-word gradient accumulation, one update per
word, shuffled words, seedable SGD/Adam) mirror nettalk/train.py so this is a
controlled loss-function swap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .constants import PHONEME_FEATURE_DIM, STRESS_FEATURE_DIM
from .data import FeatureMapping, WordDataset
from .eval import _normalize_rows, _vectorize_phoneme, _vectorize_stress
from .model import NettalkMLP, sigmoid
from .train import TrainingConfig as _UpdateConfig
from .train import _apply_update_adam, _apply_update_sgd


@dataclass(frozen=True)
class PrototypeTables:
    """Precomputed, unit-normalized prototype codes for the phoneme and stress blocks."""

    phoneme_symbols: list[str]
    phoneme_prototypes: np.ndarray  # (num_phoneme_classes, PHONEME_FEATURE_DIM), unit rows
    phoneme_index: dict[str, int]
    stress_symbols: list[str]
    stress_prototypes: np.ndarray  # (num_stress_classes, STRESS_FEATURE_DIM), unit rows
    stress_index: dict[str, int]


def build_prototype_tables(mapping: FeatureMapping) -> PrototypeTables:
    """Build normalized prototype matrices from the feature mapping.

    Uses the exact same per-symbol vectorization as the eval decoder
    (_vectorize_phoneme / _vectorize_stress) so the training targets and the
    evaluation decoder agree on what each phoneme/stress code looks like.
    """
    phoneme_symbols = sorted(mapping.phoneme_features.keys())
    phoneme_matrix = np.stack([_vectorize_phoneme(symbol, mapping) for symbol in phoneme_symbols], axis=0)
    phoneme_prototypes = _normalize_rows(phoneme_matrix)
    phoneme_index = {symbol: idx for idx, symbol in enumerate(phoneme_symbols)}

    stress_symbols = sorted(mapping.stress_encoding.keys())
    stress_matrix = np.stack([_vectorize_stress(symbol, mapping) for symbol in stress_symbols], axis=0)
    stress_prototypes = _normalize_rows(stress_matrix)
    stress_index = {symbol: idx for idx, symbol in enumerate(stress_symbols)}

    return PrototypeTables(
        phoneme_symbols=phoneme_symbols,
        phoneme_prototypes=phoneme_prototypes,
        phoneme_index=phoneme_index,
        stress_symbols=stress_symbols,
        stress_prototypes=stress_prototypes,
        stress_index=stress_index,
    )


@dataclass
class PrototypeTrainingConfig:
    """Training configuration for the prototype cross-entropy loss.

    Setting mse_weight > 0 adds a per-bit MSE anchor term against the full 26-dim
    target code, giving a hybrid loss L = cosine_CE + mse_weight * MSE. The MSE
    term supplies a dense, magnitude-carrying teaching signal (cosine-CE is
    scale-invariant, so on its own it never pins the output magnitude), while the
    cosine-CE term keeps sharpening toward the exact nearest-angle decoder.
    """

    epochs: int = 20
    tau: float = 0.1
    stress_weight: float = 1.0
    mse_weight: float = 0.0
    learning_rate: float = 1.0
    optimizer: str = "sgd"  # sgd | adam
    momentum: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    # The margin gate is an MSE-training device (it masks per-bit error below a
    # margin) and has no analogue for a softmax cross-entropy loss, so it is not
    # a field here at all.
    normalize_word_gradients: bool = False
    shuffle_words: bool = True
    seed: int = 42
    mode_name: str = "prototype_ce"


def prototype_ce_loss_and_grad(
    output_block: np.ndarray,
    prototypes_norm: np.ndarray,
    target_index: int,
    tau: float,
    eps: float = 1e-12,
) -> tuple[float, np.ndarray]:
    """Cosine-similarity softmax cross-entropy loss and gradient w.r.t. output_block.

    output_block: (D,) raw output values (e.g. post-sigmoid activations, not yet
        normalized).
    prototypes_norm: (K, D) unit-normalized prototype codes.
    target_index: index of the true class among the K prototypes.
    tau: temperature applied to the cosine logits.

    Returns (loss, dL/do) where dL/do has the same shape as output_block.
    """
    norm = float(np.linalg.norm(output_block))
    safe_norm = norm if norm > eps else eps
    o_hat = output_block / safe_norm

    cos = prototypes_norm @ o_hat  # (K,) cosine similarity to every prototype
    logits = cos / tau
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    q = exp_logits / np.sum(exp_logits)

    loss = -float(np.log(max(float(q[target_index]), 1e-300)))

    dL_dlogits = q.copy()
    dL_dlogits[target_index] -= 1.0

    # d(logit_k)/do = (1/(tau*||o||)) * (p_hat_k - (o_hat . p_hat_k) * o_hat)
    # dL/do = sum_k dL/dlogit_k * d(logit_k)/do
    weighted_proto = dL_dlogits @ prototypes_norm  # (D,)
    weighted_cos = float(np.dot(dL_dlogits, cos))
    dL_do = (weighted_proto - weighted_cos * o_hat) / (tau * safe_norm)
    return loss, dL_do


def _compute_word_gradients_prototype(
    model: NettalkMLP,
    x_word: np.ndarray,
    phoneme_word: str,
    stress_word: str,
    tables: PrototypeTables,
    config: PrototypeTrainingConfig,
    y_word: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    grad_W1 = np.zeros_like(model.W1)
    grad_b1 = np.zeros_like(model.b1)
    grad_W2 = np.zeros_like(model.W2)
    grad_b2 = np.zeros_like(model.b2)
    total_loss = 0.0

    for letter_index, x_vec in enumerate(x_word):
        hidden = sigmoid(x_vec @ model.W1 + model.b1)
        output = sigmoid(hidden @ model.W2 + model.b2)

        phoneme_target = tables.phoneme_index[phoneme_word[letter_index]]
        stress_target = tables.stress_index[stress_word[letter_index]]

        o_phoneme = output[:PHONEME_FEATURE_DIM]
        o_stress = output[PHONEME_FEATURE_DIM : PHONEME_FEATURE_DIM + STRESS_FEATURE_DIM]

        loss_phoneme, grad_phoneme = prototype_ce_loss_and_grad(
            o_phoneme, tables.phoneme_prototypes, phoneme_target, config.tau
        )
        loss_stress, grad_stress = prototype_ce_loss_and_grad(
            o_stress, tables.stress_prototypes, stress_target, config.tau
        )

        total_loss += loss_phoneme + config.stress_weight * loss_stress
        dL_do = np.concatenate([grad_phoneme, config.stress_weight * grad_stress])

        # Optional per-bit MSE anchor against the full 26-dim target code. Uses the
        # same convention as nettalk.train (loss 0.5 * sum of squared errors, so the
        # gradient is the raw per-bit error), scaled by mse_weight.
        if config.mse_weight > 0.0 and y_word is not None:
            error = output - y_word[letter_index]
            total_loss += config.mse_weight * 0.5 * float(np.sum(error**2))
            dL_do = dL_do + config.mse_weight * error

        delta_out = dL_do * output * (1.0 - output)

        delta_hidden = (model.W2 @ delta_out) * hidden * (1.0 - hidden)

        grad_W2 += np.outer(hidden, delta_out)
        grad_b2 += delta_out
        grad_W1 += np.outer(x_vec, delta_hidden)
        grad_b1 += delta_hidden

    if config.normalize_word_gradients and len(x_word) > 0:
        scale = float(len(x_word))
        grad_W1 /= scale
        grad_b1 /= scale
        grad_W2 /= scale
        grad_b2 /= scale

    return [grad_W1, grad_W2], [grad_b1, grad_b2], total_loss / max(1, len(x_word))


def train_prototype_ce(
    model: NettalkMLP,
    dataset: WordDataset,
    mapping: FeatureMapping,
    config: PrototypeTrainingConfig,
    *,
    eval_callback: Callable[[NettalkMLP], dict[str, float]] | None = None,
) -> list[dict[str, float]]:
    """Train a single-hidden-layer NettalkMLP with the prototype cross-entropy loss.

    Mirrors nettalk.train.train_model's mechanics: per-word gradient accumulation
    over the letters of a word, one weight update per word, words shuffled each
    epoch. No margin gate (it is an MSE-training device with no analogue here).
    """
    if model.hidden_dim <= 0:
        raise ValueError("train_prototype_ce requires a single-hidden-layer NettalkMLP (hidden_dim > 0).")

    tables = build_prototype_tables(mapping)
    rng = np.random.default_rng(config.seed)

    weights = [model.W1, model.W2]
    biases = [model.b1, model.b2]
    velocity_weights = [np.zeros_like(weight) for weight in weights]
    velocity_biases = [np.zeros_like(bias) for bias in biases]
    adam_m_weights = [np.zeros_like(weight) for weight in weights]
    adam_v_weights = [np.zeros_like(weight) for weight in weights]
    adam_m_biases = [np.zeros_like(bias) for bias in biases]
    adam_v_biases = [np.zeros_like(bias) for bias in biases]

    update_config = _UpdateConfig(
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        optimizer=config.optimizer,
        beta1=config.beta1,
        beta2=config.beta2,
        adam_eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, float]] = []
    step_counter = 0
    for epoch in range(1, config.epochs + 1):
        indices = np.arange(len(dataset.words))
        if config.shuffle_words:
            rng.shuffle(indices)

        epoch_loss = 0.0
        for word_index in indices:
            x_word = dataset.inputs[word_index]
            phoneme_word = dataset.phonemes[word_index]
            stress_word = dataset.stresses[word_index]
            y_word = dataset.targets[word_index]
            grad_weights, grad_biases, word_loss = _compute_word_gradients_prototype(
                model, x_word, phoneme_word, stress_word, tables, config, y_word
            )
            epoch_loss += word_loss
            step_counter += 1

            if config.optimizer == "adam":
                _apply_update_adam(
                    weights,
                    biases,
                    grad_weights,
                    grad_biases,
                    adam_m_weights,
                    adam_v_weights,
                    adam_m_biases,
                    adam_v_biases,
                    step_counter,
                    update_config,
                )
            else:
                _apply_update_sgd(
                    weights,
                    biases,
                    grad_weights,
                    grad_biases,
                    velocity_weights,
                    velocity_biases,
                    update_config,
                )

        row: dict[str, float] = {
            "epoch": float(epoch),
            "train_word_loss": float(epoch_loss / max(1, len(dataset.words))),
        }
        if eval_callback is not None:
            row.update(eval_callback(model))
        history.append(row)
    return history
