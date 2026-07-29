"""Experiment runners for baseline and modernization studies."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .constants import PHONEME_FEATURE_DIM
from .data import WordDataset
from .eval import evaluate_dataset
from .model import NettalkDeepMLP, NettalkMLP
from .train import paper_1987_config, train_model


def _merge_metric_callbacks(
    model: NettalkMLP,
    train_dataset: WordDataset,
    holdout_dataset: WordDataset,
    mapping: Any,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics.update(evaluate_dataset(model, train_dataset, mapping, prefix="train_"))
    metrics.update(evaluate_dataset(model, holdout_dataset, mapping, prefix="holdout_"))
    return metrics


def _rule_accuracy(
    model: Any,
    dataset: WordDataset,
    mapping: Any,
    *,
    letter: str,
    phoneme_symbol: str,
) -> tuple[float, int]:
    x, _, phoneme_targets, _ = dataset.flatten()
    if x.shape[0] == 0:
        return 0.0, 0

    letters = [char for word in dataset.words for char in word]
    if len(letters) != len(phoneme_targets):
        return 0.0, 0

    indices = [
        idx
        for idx, (letter_at_idx, phoneme_at_idx) in enumerate(zip(letters, phoneme_targets))
        if letter_at_idx == letter and phoneme_at_idx == phoneme_symbol
    ]
    if not indices:
        return 0.0, 0

    phoneme_symbols = sorted(set(mapping.phoneme_features.keys()) | set(phoneme_targets))
    prototypes = np.zeros((len(phoneme_symbols), PHONEME_FEATURE_DIM), dtype=np.float64)
    for idx, symbol in enumerate(phoneme_symbols):
        for feature_name in mapping.phoneme_features.get(symbol, ()):
            feature_index = mapping.feature_indices.get(feature_name)
            if feature_index is not None and 0 <= feature_index < PHONEME_FEATURE_DIM:
                prototypes[idx, feature_index] = 1.0
    proto_norm = prototypes / np.maximum(1e-12, np.linalg.norm(prototypes, axis=1, keepdims=True))

    pred = model.predict_batch(x[np.array(indices, dtype=np.int32)])[:, :PHONEME_FEATURE_DIM]
    pred_norm = pred / np.maximum(1e-12, np.linalg.norm(pred, axis=1, keepdims=True))
    scores = pred_norm @ proto_norm.T
    predicted = np.argmax(scores, axis=1)

    target_lookup = {symbol: idx for idx, symbol in enumerate(phoneme_symbols)}
    target_index = target_lookup.get(phoneme_symbol)
    if target_index is None:
        return 0.0, len(indices)
    return float(np.mean(predicted == target_index)), len(indices)


def run_hidden_scaling_experiment(
    train_dataset: WordDataset,
    holdout_dataset: WordDataset,
    mapping: Any,
    *,
    hidden_sizes: tuple[int, ...] = (0, 15, 30, 60, 120),
    epochs: int = 8,
    seed: int = 42,
) -> tuple[list[dict[str, float]], dict[int, list[dict[str, float]]]]:
    summary_rows: list[dict[str, float]] = []
    history_by_hidden: dict[int, list[dict[str, float]]] = {}

    for hidden_size in hidden_sizes:
        model = NettalkMLP(
            input_dim=train_dataset.input_dim,
            hidden_dim=hidden_size,
            seed=seed + hidden_size,
        )
        config = paper_1987_config(epochs=epochs, seed=seed + hidden_size)
        history = train_model(
            model,
            train_dataset,
            config,
            # Paper Figure 6(a) is training-set performance. Computing holdout metrics at
            # every epoch is also very expensive for dense one-hot inputs, so we only
            # compute holdout once at the end.
            eval_callback=lambda trained_model: evaluate_dataset(trained_model, train_dataset, mapping, prefix="train_"),
        )
        holdout_metrics = evaluate_dataset(model, holdout_dataset, mapping, prefix="holdout_")
        history[-1].update(holdout_metrics)
        history_by_hidden[hidden_size] = history
        final_row = dict(history[-1])
        final_row["hidden_dim"] = float(hidden_size)
        summary_rows.append(final_row)
    return summary_rows, history_by_hidden


def run_generalization_experiment(
    train_dataset: WordDataset,
    holdout_dataset: WordDataset,
    full_dataset: WordDataset,
    mapping: Any,
    *,
    hidden_dim: int = 120,
    pretrain_epochs: int = 8,
    full_passes: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    model = NettalkMLP(input_dim=train_dataset.input_dim, hidden_dim=hidden_dim, seed=seed)

    pretrain_config = paper_1987_config(epochs=pretrain_epochs, seed=seed)

    # Precompute rule counts once so they are reported alongside the learning curves.
    _, _, phoneme_targets, _ = train_dataset.flatten()
    letters = [char for word in train_dataset.words for char in word]
    hard_c_count = sum(1 for letter_at_idx, phoneme_at_idx in zip(letters, phoneme_targets) if letter_at_idx == "c" and phoneme_at_idx == "k")
    soft_c_count = sum(1 for letter_at_idx, phoneme_at_idx in zip(letters, phoneme_targets) if letter_at_idx == "c" and phoneme_at_idx == "s")

    pretrain_history = train_model(
        model,
        train_dataset,
        pretrain_config,
        eval_callback=lambda trained_model: {
            **evaluate_dataset(trained_model, train_dataset, mapping, prefix="train_"),
            "train_hard_c_accuracy": _rule_accuracy(trained_model, train_dataset, mapping, letter="c", phoneme_symbol="k")[0],
            "train_soft_c_accuracy": _rule_accuracy(trained_model, train_dataset, mapping, letter="c", phoneme_symbol="s")[0],
        },
    )

    pass_rows: list[dict[str, float]] = []
    pass_zero = {
        "dictionary_pass": 0.0,
        **evaluate_dataset(model, holdout_dataset, mapping, prefix="holdout_"),
        **evaluate_dataset(model, full_dataset, mapping, prefix="full_"),
    }
    pass_rows.append(pass_zero)

    for pass_index in range(1, full_passes + 1):
        single_pass_config = replace(pretrain_config, epochs=1, seed=seed + pass_index)
        train_model(model, full_dataset, single_pass_config, eval_callback=None)
        row = {
            "dictionary_pass": float(pass_index),
            **evaluate_dataset(model, holdout_dataset, mapping, prefix="holdout_"),
            **evaluate_dataset(model, full_dataset, mapping, prefix="full_"),
        }
        pass_rows.append(row)

    return {
        "pretrain_history": pretrain_history,
        "dictionary_pass_rows": pass_rows,
        "model": model,
        "rule_counts": {
            "hard_c": int(hard_c_count),
            "soft_c": int(soft_c_count),
        },
    }


def run_window_size_experiment(
    train_datasets: dict[int, WordDataset],
    holdout_datasets: dict[int, WordDataset],
    mapping: Any,
    *,
    hidden_dim: int = 80,
    epochs: int = 55,
    seed: int = 42,
) -> dict[str, Any]:
    history_by_window: dict[int, list[dict[str, float]]] = {}
    for window_size in sorted(train_datasets.keys()):
        train_dataset = train_datasets[window_size]
        holdout_dataset = holdout_datasets[window_size]
        model = NettalkMLP(input_dim=train_dataset.input_dim, hidden_dim=hidden_dim, seed=seed + window_size)
        config = paper_1987_config(epochs=epochs, seed=seed + window_size)
        history = train_model(
            model,
            train_dataset,
            config,
            eval_callback=lambda trained_model, td=train_dataset: evaluate_dataset(trained_model, td, mapping, prefix="train_"),
        )
        history[-1].update(evaluate_dataset(model, holdout_dataset, mapping, prefix="holdout_"))
        history_by_window[window_size] = history

    rows: list[dict[str, float]] = []
    for window_size, history in history_by_window.items():
        for row in history:
            rows.append({"window_size": float(window_size), **row})

    return {
        "rows": rows,
        "history_by_window": history_by_window,
    }


def run_two_layer_network_experiment(
    train_dataset: WordDataset,
    holdout_dataset: WordDataset,
    full_dataset: WordDataset,
    mapping: Any,
    *,
    hidden_dims: tuple[int, int] = (80, 80),
    pretrain_epochs: int = 55,
    full_passes: int = 11,
    seed: int = 42,
) -> dict[str, Any]:
    model = NettalkDeepMLP(input_dim=train_dataset.input_dim, hidden_dims=hidden_dims, seed=seed)

    pretrain_config = paper_1987_config(epochs=pretrain_epochs, seed=seed)
    pretrain_history = train_model(
        model,
        train_dataset,
        pretrain_config,
        eval_callback=lambda trained_model: evaluate_dataset(trained_model, train_dataset, mapping, prefix="train_"),
    )

    pass_rows: list[dict[str, float]] = []
    pass_rows.append(
        {
            "dictionary_pass": 0.0,
            **evaluate_dataset(model, holdout_dataset, mapping, prefix="holdout_"),
            **evaluate_dataset(model, full_dataset, mapping, prefix="full_"),
        }
    )

    for pass_index in range(1, full_passes + 1):
        single_pass_config = replace(pretrain_config, epochs=1, seed=seed + 1000 + pass_index)
        train_model(model, full_dataset, single_pass_config, eval_callback=None)
        pass_rows.append(
            {
                "dictionary_pass": float(pass_index),
                **evaluate_dataset(model, holdout_dataset, mapping, prefix="holdout_"),
                **evaluate_dataset(model, full_dataset, mapping, prefix="full_"),
            }
        )

    return {
        "pretrain_history": pretrain_history,
        "dictionary_pass_rows": pass_rows,
        "model": model,
    }


def run_damage_and_relearning_experiment(
    train_dataset: WordDataset,
    holdout_dataset: WordDataset,
    mapping: Any,
    *,
    hidden_dim: int = 80,
    pretrain_epochs: int = 8,
    damage_levels: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
    relearn_damage: float = 0.3,
    relearn_epochs: int = 6,
    seed: int = 42,
) -> dict[str, Any]:
    baseline = NettalkMLP(input_dim=train_dataset.input_dim, hidden_dim=hidden_dim, seed=seed)
    baseline_config = paper_1987_config(epochs=pretrain_epochs, seed=seed)
    baseline_history = train_model(
        baseline,
        train_dataset,
        baseline_config,
        eval_callback=lambda model: evaluate_dataset(model, holdout_dataset, mapping, prefix="holdout_"),
    )

    damage_rows: list[dict[str, float]] = []
    damage_template = _sample_uniform_noise_template(baseline, np.random.default_rng(seed + 991))
    for damage in damage_levels:
        damaged = baseline.copy()
        _apply_noise_template(damaged, damage_template, damage)
        row = {"damage_level": float(damage)}
        row.update(evaluate_dataset(damaged, holdout_dataset, mapping, prefix="holdout_"))
        damage_rows.append(row)

    relearn_model = baseline.copy()
    _apply_noise_template(relearn_model, damage_template, relearn_damage)
    relearn_config = paper_1987_config(epochs=relearn_epochs, seed=seed + 1001)
    relearn_history = train_model(
        relearn_model,
        train_dataset,
        relearn_config,
        eval_callback=lambda model: evaluate_dataset(model, holdout_dataset, mapping, prefix="holdout_"),
    )

    fresh_model = NettalkMLP(input_dim=train_dataset.input_dim, hidden_dim=hidden_dim, seed=seed + 1002)
    fresh_history = train_model(
        fresh_model,
        train_dataset,
        relearn_config,
        eval_callback=lambda model: evaluate_dataset(model, holdout_dataset, mapping, prefix="holdout_"),
    )

    return {
        "baseline_history": baseline_history,
        "damage_rows": damage_rows,
        "relearn_history": relearn_history,
        "fresh_history": fresh_history,
    }


def _sample_old_word_sequence(old_dataset: WordDataset, total_words: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    old_indices = np.arange(len(old_dataset.words))
    if len(old_indices) == 0 or total_words <= 0:
        return []
    return [int(rng.choice(old_indices)) for _ in range(total_words)]


def _build_schedule_dataset(
    old_dataset: WordDataset,
    new_dataset: WordDataset,
    *,
    repeats: int,
    old_sequence: list[int],
    interleaved: bool,
) -> WordDataset:
    repeat_count = max(1, int(repeats))
    novel_sequence = [index for _ in range(repeat_count) for index in range(len(new_dataset.words))]
    if len(old_sequence) != len(novel_sequence):
        raise ValueError("Old-word sequence must match the number of novel presentations.")

    words: list[str] = []
    phonemes: list[str] = []
    stresses: list[str] = []
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    def append_word(dataset: WordDataset, index: int) -> None:
        words.append(dataset.words[index])
        phonemes.append(dataset.phonemes[index])
        stresses.append(dataset.stresses[index])
        inputs.append(dataset.inputs[index].copy())
        targets.append(dataset.targets[index].copy())

    if interleaved:
        for novel_index, old_index in zip(novel_sequence, old_sequence):
            append_word(new_dataset, novel_index)
            append_word(old_dataset, old_index)
    else:
        for novel_index in novel_sequence:
            append_word(new_dataset, novel_index)
        for old_index in old_sequence:
            append_word(old_dataset, old_index)

    return WordDataset(
        words=words,
        phonemes=phonemes,
        stresses=stresses,
        inputs=inputs,
        targets=targets,
        window_size=new_dataset.window_size,
        name="interleaved_schedule" if interleaved else "massed_schedule",
    )


def _build_spacing_schedules(
    old_dataset: WordDataset,
    new_dataset: WordDataset,
    *,
    repeats: int,
    seed: int,
) -> tuple[WordDataset, WordDataset]:
    repeat_count = max(1, int(repeats))
    novel_presentations = len(new_dataset.words) * repeat_count
    old_sequence = _sample_old_word_sequence(old_dataset, novel_presentations, seed=seed)
    massed_dataset = _build_schedule_dataset(
        old_dataset,
        new_dataset,
        repeats=repeat_count,
        old_sequence=old_sequence,
        interleaved=False,
    )
    interleaved_dataset = _build_schedule_dataset(
        old_dataset,
        new_dataset,
        repeats=repeat_count,
        old_sequence=old_sequence,
        interleaved=True,
    )
    return massed_dataset, interleaved_dataset


def run_spacing_experiment(
    old_dataset: WordDataset,
    holdout_dataset: WordDataset,
    mapping: Any,
    *,
    hidden_dim: int = 80,
    novel_words: int = 250,
    repeats: int = 3,
    pretrain_epochs: int = 6,
    seed: int = 42,
) -> dict[str, Any]:
    novel_count = max(10, min(novel_words, len(holdout_dataset.words)))
    novel_dataset = holdout_dataset.subset(list(range(novel_count)), name="novel_subset")
    massed_dataset, interleaved_dataset = _build_spacing_schedules(
        old_dataset,
        novel_dataset,
        repeats=repeats,
        seed=seed,
    )

    base_model = NettalkMLP(input_dim=old_dataset.input_dim, hidden_dim=hidden_dim, seed=seed)
    pretrain_config = paper_1987_config(epochs=pretrain_epochs, seed=seed)
    train_model(base_model, old_dataset, pretrain_config)

    massed_model = base_model.copy()
    interleaved_model = base_model.copy()

    schedule_config = paper_1987_config(epochs=1, seed=seed + 77)
    schedule_config.shuffle_words = False
    train_model(massed_model, massed_dataset, schedule_config)
    train_model(interleaved_model, interleaved_dataset, schedule_config)

    massed_metrics = evaluate_dataset(massed_model, novel_dataset, mapping, prefix="novel_")
    interleaved_metrics = evaluate_dataset(interleaved_model, novel_dataset, mapping, prefix="novel_")

    return {
        "novel_word_count": float(novel_count),
        "repeats": float(repeats),
        "novel_presentations": float(len(novel_dataset.words) * max(1, int(repeats))),
        "old_rehearsal_presentations": float(len(massed_dataset.words) - len(novel_dataset.words) * max(1, int(repeats))),
        "schedule_total_updates": float(len(massed_dataset.words)),
        "schedule_shuffle_words": False,
        "massed_novel_metrics": massed_metrics,
        "interleaved_novel_metrics": interleaved_metrics,
    }


def fit_power_law(history: list[dict[str, float]], *, x_key: str = "epoch", y_key: str = "train_word_mse") -> dict[str, float]:
    if len(history) < 3:
        return {"intercept": 0.0, "exponent": 0.0, "r2": 0.0}

    x = np.array([row[x_key] for row in history], dtype=np.float64)
    y = np.array([row[y_key] for row in history], dtype=np.float64)
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return {"intercept": 0.0, "exponent": 0.0, "r2": 0.0}

    log_x = np.log(x)
    log_y = np.log(y)
    exponent, intercept = np.polyfit(log_x, log_y, deg=1)
    pred = intercept + exponent * log_x
    ss_res = float(np.sum((log_y - pred) ** 2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r2 = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return {
        "intercept": float(intercept),
        "exponent": float(exponent),
        "r2": float(r2),
    }


def _model_parameter_arrays(model: NettalkMLP) -> dict[str, np.ndarray]:
    if model.hidden_dim > 0:
        return {
            "W1": model.W1,
            "b1": model.b1,
            "W2": model.W2,
            "b2": model.b2,
        }
    return {
        "W": model.W,
        "b": model.b,
    }


def _sample_uniform_noise_template(model: NettalkMLP, rng: np.random.Generator) -> dict[str, np.ndarray]:
    params = _model_parameter_arrays(model)
    return {name: rng.uniform(-1.0, 1.0, size=value.shape) for name, value in params.items()}


def _apply_noise_template(model: NettalkMLP, template: dict[str, np.ndarray], scale: float) -> None:
    if scale <= 0:
        return
    params = _model_parameter_arrays(model)
    for name, value in params.items():
        value += scale * template[name]


def _symmetric_quantize(array: np.ndarray, bits: int) -> np.ndarray:
    if bits <= 0:
        return array.copy()
    levels = max(2, int(2**bits))
    max_abs = float(np.max(np.abs(array)))
    if max_abs < 1e-12:
        return array.copy()
    step = (2.0 * max_abs) / float(levels - 1)
    quantized = np.round((array + max_abs) / step) * step - max_abs
    return np.clip(quantized, -max_abs, max_abs)


def run_quantization_experiment(
    trained_model: NettalkMLP,
    holdout_dataset: WordDataset,
    mapping: Any,
    *,
    bits_list: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 12),
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []

    baseline_metrics = evaluate_dataset(trained_model, holdout_dataset, mapping, prefix="holdout_")
    baseline_row = {"bits": 64.0}
    baseline_row.update(baseline_metrics)
    rows.append(baseline_row)

    for bits in bits_list:
        quantized = trained_model.copy()
        params = _model_parameter_arrays(quantized)
        for name, value in params.items():
            params[name][...] = _symmetric_quantize(value, bits)

        row = {"bits": float(bits)}
        row.update(evaluate_dataset(quantized, holdout_dataset, mapping, prefix="holdout_"))
        rows.append(row)
    return rows


def _pairwise_euclidean(vectors: np.ndarray) -> np.ndarray:
    diffs = vectors[:, None, :] - vectors[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=2))


def _complete_linkage_distance(cluster_a: list[int], cluster_b: list[int], distance_matrix: np.ndarray) -> float:
    max_distance = 0.0
    for idx_a in cluster_a:
        for idx_b in cluster_b:
            distance = float(distance_matrix[idx_a, idx_b])
            if distance > max_distance:
                max_distance = distance
    return max_distance


def _complete_linkage_clustering(distance_matrix: np.ndarray) -> tuple[list[dict[str, float]], list[int]]:
    cluster_members: dict[int, list[int]] = {idx: [idx] for idx in range(distance_matrix.shape[0])}
    children: dict[int, tuple[int, int]] = {}
    active_ids = list(cluster_members.keys())
    next_cluster_id = distance_matrix.shape[0]
    merges: list[dict[str, float]] = []

    while len(active_ids) > 1:
        best_pair: tuple[int, int] | None = None
        best_distance = float("inf")
        for left_position in range(len(active_ids)):
            left_id = active_ids[left_position]
            for right_position in range(left_position + 1, len(active_ids)):
                right_id = active_ids[right_position]
                distance = _complete_linkage_distance(
                    cluster_members[left_id],
                    cluster_members[right_id],
                    distance_matrix,
                )
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (left_id, right_id)

        if best_pair is None:
            break

        left_id, right_id = best_pair
        merged_members = cluster_members[left_id] + cluster_members[right_id]
        cluster_members[next_cluster_id] = merged_members
        children[next_cluster_id] = (left_id, right_id)
        merges.append(
            {
                "left_id": float(left_id),
                "right_id": float(right_id),
                "distance": float(best_distance),
                "merged_size": float(len(merged_members)),
                "new_id": float(next_cluster_id),
            }
        )

        active_ids = [cluster_id for cluster_id in active_ids if cluster_id not in (left_id, right_id)]
        active_ids.append(next_cluster_id)
        next_cluster_id += 1

    if not active_ids:
        return merges, []

    root_id = active_ids[0]

    def _leaf_order(cluster_id: int) -> list[int]:
        if cluster_id < distance_matrix.shape[0]:
            return [cluster_id]
        left_child, right_child = children[cluster_id]
        return _leaf_order(left_child) + _leaf_order(right_child)

    return merges, _leaf_order(root_id)


def run_hidden_clustering_experiment(
    trained_model: NettalkMLP,
    dataset: WordDataset,
    *,
    max_letters: int = 60000,
) -> dict[str, Any]:
    if trained_model.hidden_dim <= 0:
        return {
            "labels": [],
            "distance_matrix_ordered": [],
            "merge_rows": [],
            "leaf_order": [],
            "note": "hidden_dim_is_zero",
        }

    if not dataset.words:
        return {
            "labels": [],
            "distance_matrix_ordered": [],
            "merge_rows": [],
            "leaf_order": [],
            "note": "empty_dataset",
        }

    # Paper method (1987): average hidden activity vectors for each letter-to-sound correspondence.
    # Here we interpret correspondence as (center letter, center phoneme symbol).
    correspondence_sums: dict[str, np.ndarray] = {}
    correspondence_counts: dict[str, int] = {}
    total_letters = 0

    for word, phonemes, x_word in zip(dataset.words, dataset.phonemes, dataset.inputs):
        if len(word) == 0:
            continue
        hidden_word = trained_model.hidden_batch(x_word)
        for position, (letter, phoneme_symbol) in enumerate(zip(word, phonemes)):
            total_letters += 1
            if total_letters > max_letters:
                break
            key = f"{letter}->{phoneme_symbol}"
            if key not in correspondence_sums:
                correspondence_sums[key] = np.zeros((trained_model.hidden_dim,), dtype=np.float64)
                correspondence_counts[key] = 0
            correspondence_sums[key] += hidden_word[position]
            correspondence_counts[key] += 1
        if total_letters > max_letters:
            break

    labels = sorted(correspondence_sums.keys())
    prototypes = np.zeros((len(labels), trained_model.hidden_dim), dtype=np.float64)
    counts = np.zeros((len(labels),), dtype=np.int32)
    for label_index, label in enumerate(labels):
        count = max(1, correspondence_counts[label])
        prototypes[label_index] = correspondence_sums[label] / float(count)
        counts[label_index] = count

    distance_matrix = _pairwise_euclidean(prototypes)
    merge_rows, leaf_order = _complete_linkage_clustering(distance_matrix)
    ordered_labels = [labels[index] for index in leaf_order]
    ordered_distance = distance_matrix[np.ix_(leaf_order, leaf_order)]

    return {
        "labels": ordered_labels,
        "distance_matrix_ordered": ordered_distance.tolist(),
        "merge_rows": merge_rows,
        "leaf_order": [float(index) for index in leaf_order],
        "sampled_letters": float(min(total_letters, max_letters)),
        "correspondence_counts": {label: int(correspondence_counts[label]) for label in labels},
    }
