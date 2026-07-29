"""Evaluation metrics for NETtalk reproduction."""

from __future__ import annotations

import numpy as np

from .constants import PHONEME_FEATURE_DIM, STRESS_FEATURE_DIM
from .data import FeatureMapping, WordDataset
from .model import NettalkMLP


def _vectorize_phoneme(symbol: str, mapping: FeatureMapping) -> np.ndarray:
    vector = np.zeros((PHONEME_FEATURE_DIM,), dtype=np.float64)
    for feature_name in mapping.phoneme_features.get(symbol, ()):
        index = mapping.feature_indices.get(feature_name)
        if index is not None and 0 <= index < PHONEME_FEATURE_DIM:
            vector[index] = 1.0
    return vector


def _vectorize_stress(symbol: str, mapping: FeatureMapping) -> np.ndarray:
    order = [name for name, _ in sorted(mapping.stress_indices.items(), key=lambda pair: pair[1])]
    index_lookup = {name: idx for idx, name in enumerate(order)}
    vector = np.zeros((STRESS_FEATURE_DIM,), dtype=np.float64)
    for feature_name in mapping.stress_encoding.get(symbol, ()):
        if feature_name in index_lookup:
            vector[index_lookup[feature_name]] = 1.0
    return vector


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    return matrix / safe_norms


def _best_guess_accuracy(
    pred: np.ndarray,
    target_symbols: list[str],
    prototype_symbols: list[str],
    prototype_matrix: np.ndarray,
) -> float:
    if pred.shape[0] == 0:
        return 0.0
    symbol_to_index = {symbol: idx for idx, symbol in enumerate(prototype_symbols)}
    target_indices = np.array([symbol_to_index[symbol] for symbol in target_symbols], dtype=np.int32)

    pred_norm = _normalize_rows(pred)
    proto_norm = _normalize_rows(prototype_matrix)
    scores = pred_norm @ proto_norm.T
    predicted_indices = np.argmax(scores, axis=1)
    return float(np.mean(predicted_indices == target_indices))


def evaluate_dataset(
    model: NettalkMLP,
    dataset: WordDataset,
    mapping: FeatureMapping,
    *,
    margin: float = 0.1,
    prefix: str = "",
) -> dict[str, float]:
    x, y_true, phoneme_targets, stress_targets = dataset.flatten()
    if len(x) == 0:
        return {f"{prefix}mse": 0.0}

    y_pred = model.predict_batch(x)
    mse = float(np.mean((y_pred - y_true) ** 2))

    # "Best guess" in the NETtalk benchmark is defined by choosing the closest
    # (smallest angle) phoneme code among the phoneme inventory, not just among
    # the labels present in a specific subset.
    phoneme_symbols = sorted(set(mapping.phoneme_features.keys()) | set(phoneme_targets))
    phoneme_prototypes = np.stack([_vectorize_phoneme(symbol, mapping) for symbol in phoneme_symbols], axis=0)
    stress_symbols = sorted(set(mapping.stress_encoding.keys()) | set(stress_targets))
    stress_prototypes = np.stack([_vectorize_stress(symbol, mapping) for symbol in stress_symbols], axis=0)

    phoneme_best_guess = _best_guess_accuracy(
        y_pred[:, :PHONEME_FEATURE_DIM],
        phoneme_targets,
        phoneme_symbols,
        phoneme_prototypes,
    )
    stress_best_guess = _best_guess_accuracy(
        y_pred[:, PHONEME_FEATURE_DIM : PHONEME_FEATURE_DIM + STRESS_FEATURE_DIM],
        stress_targets,
        stress_symbols,
        stress_prototypes,
    )

    perfect_match_phoneme = np.abs(y_pred[:, :PHONEME_FEATURE_DIM] - y_true[:, :PHONEME_FEATURE_DIM]) <= margin
    perfect_match_all = np.abs(y_pred - y_true) <= margin

    return {
        f"{prefix}mse": mse,
        f"{prefix}phoneme_best_guess": phoneme_best_guess,
        f"{prefix}stress_best_guess": stress_best_guess,
        f"{prefix}perfect_match_phoneme": float(np.mean(np.all(perfect_match_phoneme, axis=1))),
        f"{prefix}perfect_match_all": float(np.mean(np.all(perfect_match_all, axis=1))),
    }
