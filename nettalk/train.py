"""Training loops for NETtalk baseline and ablations.

The original NETtalk implementation trained with per-word updates:
- accumulate gradients across letters in a word
- apply one weight update per word

This module implements that mechanic for both the single-hidden-layer baseline
and a small deep variant used for the 1987 "extra hidden layer" experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .data import WordDataset
from .model import NettalkDeepMLP, NettalkMLP, sigmoid

Model = NettalkMLP | NettalkDeepMLP


@dataclass
class TrainingConfig:
    """Configurable training behavior for baseline and modernization runs."""

    epochs: int = 8
    learning_rate: float = 1.0
    momentum: float = 0.0
    optimizer: str = "sgd"  # sgd | adam
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    use_margin_gate: bool = True
    margin: float = 0.1
    normalize_word_gradients: bool = True
    shuffle_words: bool = True
    seed: int = 42
    mode_name: str = "paper_1987"


def paper_1987_config(*, epochs: int = 8, seed: int = 42) -> TrainingConfig:
    return TrainingConfig(
        epochs=epochs,
        learning_rate=1.0,
        momentum=0.0,
        optimizer="sgd",
        use_margin_gate=True,
        margin=0.1,
        # The paper describes accumulating gradients over all letters of a word and
        # applying a single update once per word. It does not mention averaging by
        # word length, so we interpret this literally as a sum.
        normalize_word_gradients=False,
        seed=seed,
        mode_name="paper_1987",
    )


def paper_1986_config(*, epochs: int = 8, seed: int = 42) -> TrainingConfig:
    return TrainingConfig(
        epochs=epochs,
        learning_rate=2.0,
        momentum=0.9,
        optimizer="sgd",
        use_margin_gate=True,
        margin=0.1,
        normalize_word_gradients=False,
        seed=seed,
        mode_name="paper_1986",
    )


def _weights_and_biases(model: Model) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if isinstance(model, NettalkDeepMLP):
        return model.weights, model.biases
    if model.hidden_dim > 0:
        return [model.W1, model.W2], [model.b1, model.b2]
    return [model.W], [model.b]


def _compute_word_gradients(
    model: Model,
    x_word: np.ndarray,
    y_word: np.ndarray,
    config: TrainingConfig,
) -> tuple[list[np.ndarray], list[np.ndarray], float, float]:
    weights, biases = _weights_and_biases(model)
    grad_weights = [np.zeros_like(weight) for weight in weights]
    grad_biases = [np.zeros_like(bias) for bias in biases]
    total_loss = 0.0
    gated_units = 0
    total_units = 0

    for x_vec, y_true in zip(x_word, y_word):
        activations: list[np.ndarray] = [x_vec]
        for weight, bias in zip(weights, biases):
            activations.append(sigmoid(activations[-1] @ weight + bias))
        y_pred = activations[-1]
        error = y_pred - y_true
        total_loss += 0.5 * float(np.mean(error**2))

        if config.use_margin_gate:
            gate = (np.abs(error) > config.margin).astype(np.float64)
            gated_units += int(gate.sum())
            total_units += int(gate.size)
        else:
            gate = np.ones_like(error)
            gated_units += int(gate.size)
            total_units += int(gate.size)

        delta_out = error * y_pred * (1.0 - y_pred) * gate

        deltas: list[np.ndarray | None] = [None] * (len(weights) + 1)
        deltas[-1] = delta_out
        for layer_index in range(len(weights) - 1, 0, -1):
            delta_next = deltas[layer_index + 1]
            assert delta_next is not None
            hidden_activation = activations[layer_index]
            deltas[layer_index] = (weights[layer_index] @ delta_next) * hidden_activation * (1.0 - hidden_activation)

        for layer_index in range(len(weights)):
            delta_layer = deltas[layer_index + 1]
            assert delta_layer is not None
            grad_weights[layer_index] += np.outer(activations[layer_index], delta_layer)
            grad_biases[layer_index] += delta_layer

    if config.normalize_word_gradients and len(x_word) > 0:
        scale = float(len(x_word))
        for gradient in grad_weights:
            gradient /= scale
        for gradient in grad_biases:
            gradient /= scale

    gated_fraction = float(gated_units / max(1, total_units))
    return grad_weights, grad_biases, total_loss / max(1, len(x_word)), gated_fraction


def _apply_update_sgd(
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    grad_weights: list[np.ndarray],
    grad_biases: list[np.ndarray],
    velocity_weights: list[np.ndarray],
    velocity_biases: list[np.ndarray],
    config: TrainingConfig,
) -> None:
    lr = config.learning_rate

    for index, parameter in enumerate(weights):
        grad = grad_weights[index]
        if config.weight_decay > 0:
            grad = grad + config.weight_decay * parameter
        velocity_weights[index] = config.momentum * velocity_weights[index] - lr * grad
        parameter += velocity_weights[index]

    for index, parameter in enumerate(biases):
        grad = grad_biases[index]
        if config.weight_decay > 0:
            grad = grad + config.weight_decay * parameter
        velocity_biases[index] = config.momentum * velocity_biases[index] - lr * grad
        parameter += velocity_biases[index]


def _apply_update_adam(
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    grad_weights: list[np.ndarray],
    grad_biases: list[np.ndarray],
    adam_m_weights: list[np.ndarray],
    adam_v_weights: list[np.ndarray],
    adam_m_biases: list[np.ndarray],
    adam_v_biases: list[np.ndarray],
    step: int,
    config: TrainingConfig,
) -> None:
    lr = config.learning_rate
    beta1 = config.beta1
    beta2 = config.beta2
    eps = config.adam_eps

    for index, parameter in enumerate(weights):
        grad = grad_weights[index]
        if config.weight_decay > 0:
            grad = grad + config.weight_decay * parameter
        adam_m_weights[index] = beta1 * adam_m_weights[index] + (1.0 - beta1) * grad
        adam_v_weights[index] = beta2 * adam_v_weights[index] + (1.0 - beta2) * (grad * grad)
        m_hat = adam_m_weights[index] / (1.0 - beta1**step)
        v_hat = adam_v_weights[index] / (1.0 - beta2**step)
        parameter -= lr * m_hat / (np.sqrt(v_hat) + eps)

    for index, parameter in enumerate(biases):
        grad = grad_biases[index]
        if config.weight_decay > 0:
            grad = grad + config.weight_decay * parameter
        adam_m_biases[index] = beta1 * adam_m_biases[index] + (1.0 - beta1) * grad
        adam_v_biases[index] = beta2 * adam_v_biases[index] + (1.0 - beta2) * (grad * grad)
        m_hat = adam_m_biases[index] / (1.0 - beta1**step)
        v_hat = adam_v_biases[index] / (1.0 - beta2**step)
        parameter -= lr * m_hat / (np.sqrt(v_hat) + eps)


def train_model(
    model: Model,
    dataset: WordDataset,
    config: TrainingConfig,
    *,
    eval_callback: Callable[[Model], dict[str, float]] | None = None,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, float]] = []
    weights, biases = _weights_and_biases(model)
    velocity_weights = [np.zeros_like(weight) for weight in weights]
    velocity_biases = [np.zeros_like(bias) for bias in biases]
    adam_m_weights = [np.zeros_like(weight) for weight in weights]
    adam_v_weights = [np.zeros_like(weight) for weight in weights]
    adam_m_biases = [np.zeros_like(bias) for bias in biases]
    adam_v_biases = [np.zeros_like(bias) for bias in biases]

    step_counter = 0
    for epoch in range(1, config.epochs + 1):
        indices = np.arange(len(dataset.words))
        if config.shuffle_words:
            rng.shuffle(indices)

        epoch_loss = 0.0
        gate_fraction_accumulator = 0.0
        for index in indices:
            x_word = dataset.inputs[index]
            y_word = dataset.targets[index]
            grad_weights, grad_biases, word_loss, gate_fraction = _compute_word_gradients(model, x_word, y_word, config)
            epoch_loss += word_loss
            gate_fraction_accumulator += gate_fraction
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
                    config,
                )
            else:
                _apply_update_sgd(
                    weights,
                    biases,
                    grad_weights,
                    grad_biases,
                    velocity_weights,
                    velocity_biases,
                    config,
                )

        row: dict[str, float] = {
            "epoch": float(epoch),
            "train_word_mse": float(epoch_loss / max(1, len(dataset.words))),
            "gated_fraction": float(gate_fraction_accumulator / max(1, len(dataset.words))),
        }
        if eval_callback is not None:
            row.update(eval_callback(model))
        history.append(row)
    return history
