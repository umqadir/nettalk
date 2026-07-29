"""Baseline NETtalk MLP model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import OUTPUT_DIM


def sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class NettalkMLP:
    """Small fully-connected MLP used by the NETtalk baseline."""

    input_dim: int
    hidden_dim: int
    output_dim: int = OUTPUT_DIM
    init_range: float = 0.3
    seed: int = 42

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        if self.hidden_dim > 0:
            self.W1 = rng.uniform(-self.init_range, self.init_range, size=(self.input_dim, self.hidden_dim))
            self.b1 = rng.uniform(-self.init_range, self.init_range, size=(self.hidden_dim,))
            self.W2 = rng.uniform(-self.init_range, self.init_range, size=(self.hidden_dim, self.output_dim))
            self.b2 = rng.uniform(-self.init_range, self.init_range, size=(self.output_dim,))
        else:
            self.W = rng.uniform(-self.init_range, self.init_range, size=(self.input_dim, self.output_dim))
            self.b = rng.uniform(-self.init_range, self.init_range, size=(self.output_dim,))

    def copy(self) -> "NettalkMLP":
        replica = NettalkMLP(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            init_range=self.init_range,
            seed=self.seed,
        )
        if self.hidden_dim > 0:
            replica.W1 = self.W1.copy()
            replica.b1 = self.b1.copy()
            replica.W2 = self.W2.copy()
            replica.b2 = self.b2.copy()
        else:
            replica.W = self.W.copy()
            replica.b = self.b.copy()
        return replica

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray | None, np.ndarray]]:
        if self.hidden_dim > 0:
            hidden = sigmoid(x @ self.W1 + self.b1)
            output = sigmoid(hidden @ self.W2 + self.b2)
            return output, (hidden, x)
        output = sigmoid(x @ self.W + self.b)
        return output, (None, x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.hidden_dim > 0:
            hidden = sigmoid(x @ self.W1 + self.b1)
            return sigmoid(hidden @ self.W2 + self.b2)
        return sigmoid(x @ self.W + self.b)

    def predict_batch(self, x: np.ndarray) -> np.ndarray:
        if self.hidden_dim > 0:
            hidden = sigmoid(x @ self.W1 + self.b1)
            return sigmoid(hidden @ self.W2 + self.b2)
        return sigmoid(x @ self.W + self.b)

    def hidden_batch(self, x: np.ndarray) -> np.ndarray:
        if self.hidden_dim > 0:
            return sigmoid(x @ self.W1 + self.b1)
        return np.zeros((x.shape[0], 0), dtype=np.float64)

    def add_uniform_noise(self, scale: float, rng: np.random.Generator) -> None:
        if scale <= 0:
            return
        if self.hidden_dim > 0:
            self.W1 += rng.uniform(-scale, scale, size=self.W1.shape)
            self.b1 += rng.uniform(-scale, scale, size=self.b1.shape)
            self.W2 += rng.uniform(-scale, scale, size=self.W2.shape)
            self.b2 += rng.uniform(-scale, scale, size=self.b2.shape)
        else:
            self.W += rng.uniform(-scale, scale, size=self.W.shape)
            self.b += rng.uniform(-scale, scale, size=self.b.shape)


@dataclass
class NettalkDeepMLP:
    """Fully-connected MLP with an arbitrary number of hidden layers."""

    input_dim: int
    hidden_dims: tuple[int, ...]
    output_dim: int = OUTPUT_DIM
    init_range: float = 0.3
    seed: int = 42

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        layer_dims = [self.input_dim, *self.hidden_dims, self.output_dim]
        self.weights = [
            rng.uniform(-self.init_range, self.init_range, size=(layer_dims[idx], layer_dims[idx + 1]))
            for idx in range(len(layer_dims) - 1)
        ]
        self.biases = [
            rng.uniform(-self.init_range, self.init_range, size=(layer_dims[idx + 1],))
            for idx in range(len(layer_dims) - 1)
        ]

    def copy(self) -> "NettalkDeepMLP":
        replica = NettalkDeepMLP(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            init_range=self.init_range,
            seed=self.seed,
        )
        replica.weights = [weight.copy() for weight in self.weights]
        replica.biases = [bias.copy() for bias in self.biases]
        return replica

    def predict_batch(self, x: np.ndarray) -> np.ndarray:
        activation = x
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            activation = sigmoid(activation @ weight + bias)
        return sigmoid(activation @ self.weights[-1] + self.biases[-1])

    def add_uniform_noise(self, scale: float, rng: np.random.Generator) -> None:
        if scale <= 0:
            return
        for idx in range(len(self.weights)):
            self.weights[idx] += rng.uniform(-scale, scale, size=self.weights[idx].shape)
            self.biases[idx] += rng.uniform(-scale, scale, size=self.biases[idx].shape)
