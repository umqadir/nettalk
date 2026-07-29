"""Plotting utilities for NETtalk experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _prepare_path(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_hidden_scaling(rows: list[dict[str, float]], path: str | Path) -> None:
    output_path = _prepare_path(path)
    rows_sorted = sorted(rows, key=lambda item: item["hidden_dim"])
    hidden = [row["hidden_dim"] for row in rows_sorted]
    train_acc = [100.0 * row.get("train_phoneme_best_guess", 0.0) for row in rows_sorted]
    holdout_acc = [100.0 * row.get("holdout_phoneme_best_guess", 0.0) for row in rows_sorted]

    plt.figure(figsize=(8, 5))
    plt.plot(hidden, train_acc, marker="o", label="Train (1000 words)")
    plt.plot(hidden, holdout_acc, marker="o", label="Holdout dictionary")
    plt.xlabel("Hidden units")
    plt.ylabel("Best-guess phoneme accuracy (%)")
    plt.title("NETtalk hidden-unit scaling")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_scaling_histories(history_by_hidden: dict[int, list[dict[str, float]]], path: str | Path) -> None:
    output_path = _prepare_path(path)
    plt.figure(figsize=(8, 5))
    for hidden_dim in sorted(history_by_hidden.keys()):
        rows = history_by_hidden[hidden_dim]
        x = [row["epoch"] for row in rows]
        y = [100.0 * row.get("train_phoneme_best_guess", 0.0) for row in rows]
        plt.plot(x, y, marker="o", label=f"H={hidden_dim}")

    plt.xlabel("Epoch")
    plt.ylabel("Train best-guess phoneme accuracy (%)")
    plt.title("Scaling learning curves by hidden size (1000-word train)")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_window_size_learning(history_by_window: dict[int, list[dict[str, float]]], path: str | Path) -> None:
    output_path = _prepare_path(path)
    plt.figure(figsize=(8, 5))
    for window_size in sorted(history_by_window.keys()):
        rows = history_by_window[window_size]
        x = [row["epoch"] for row in rows]
        y = [100.0 * row.get("train_phoneme_best_guess", 0.0) for row in rows]
        plt.plot(x, y, marker="o", label=f"w={window_size}")

    plt.xlabel("Epoch (passes through 1000-word corpus)")
    plt.ylabel("Train best-guess phoneme accuracy (%)")
    plt.title("Context window size vs learning speed (H=80)")
    plt.grid(alpha=0.3)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_generalization(rows: list[dict[str, float]], path: str | Path) -> None:
    output_path = _prepare_path(path)
    passes = [row["dictionary_pass"] for row in rows]
    holdout = [100.0 * row.get("holdout_phoneme_best_guess", 0.0) for row in rows]
    full = [100.0 * row.get("full_phoneme_best_guess", 0.0) for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(passes, holdout, marker="o", label="Holdout dictionary")
    plt.plot(passes, full, marker="o", label="Full dictionary")
    plt.xlabel("Dictionary passes after 1000-word pretraining")
    plt.ylabel("Best-guess phoneme accuracy (%)")
    plt.title("Dictionary generalization and continued training")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_pretrain_history(rows: list[dict[str, float]], path: str | Path) -> None:
    output_path = _prepare_path(path)
    epochs = [row["epoch"] for row in rows]
    train_acc = [100.0 * row.get("train_phoneme_best_guess", 0.0) for row in rows]
    has_holdout = any("holdout_phoneme_best_guess" in row for row in rows)
    holdout_acc = [100.0 * row.get("holdout_phoneme_best_guess", 0.0) for row in rows] if has_holdout else []
    train_mse = [row.get("train_word_mse", 0.0) for row in rows]

    fig, axis_left = plt.subplots(figsize=(8, 5))
    axis_left.plot(epochs, train_acc, marker="o", label="Train best-guess")
    if has_holdout:
        axis_left.plot(epochs, holdout_acc, marker="o", label="Holdout best-guess")
    axis_left.set_xlabel("Pretraining epoch")
    axis_left.set_ylabel("Best-guess phoneme accuracy (%)")
    axis_left.grid(alpha=0.3)

    axis_right = axis_left.twinx()
    axis_right.plot(epochs, train_mse, color="tab:red", linestyle="--", marker="x", label="Train word MSE")
    axis_right.set_ylabel("Train word MSE")

    lines = axis_left.get_lines() + axis_right.get_lines()
    labels = [line.get_label() for line in lines]
    axis_left.legend(lines, labels, loc="best")
    plt.title("Pretraining dynamics (1000-word subset)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_rule_learning(rows: list[dict[str, float]], path: str | Path) -> None:
    output_path = _prepare_path(path)
    epochs = [row["epoch"] for row in rows]
    hard_acc = [100.0 * row.get("train_hard_c_accuracy", 0.0) for row in rows]
    soft_acc = [100.0 * row.get("train_soft_c_accuracy", 0.0) for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, hard_acc, marker="o", label="Hard c (c->k)")
    plt.plot(epochs, soft_acc, marker="o", label="Soft c (c->s)")
    plt.xlabel("Epoch (passes through 1000-word corpus)")
    plt.ylabel("Accuracy (%) on correspondence instances")
    plt.title("Learning curves for hard vs soft 'c' correspondences")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_generalization_comparison(
    wide_rows: list[dict[str, float]],
    deep_rows: list[dict[str, float]],
    path: str | Path,
    *,
    wide_label: str = "One hidden layer (H=120)",
    deep_label: str = "Two hidden layers (80,80)",
) -> None:
    output_path = _prepare_path(path)
    wide_passes = [row["dictionary_pass"] for row in wide_rows]
    wide_full = [100.0 * row.get("full_phoneme_best_guess", 0.0) for row in wide_rows]
    deep_passes = [row["dictionary_pass"] for row in deep_rows]
    deep_full = [100.0 * row.get("full_phoneme_best_guess", 0.0) for row in deep_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(wide_passes, wide_full, marker="o", label=wide_label)
    plt.plot(deep_passes, deep_full, marker="o", label=deep_label)
    plt.xlabel("Dictionary passes after 1000-word pretraining")
    plt.ylabel("Full dictionary best-guess phoneme accuracy (%)")
    plt.title("Generalization: wide vs deep architectures")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_damage(rows: list[dict[str, float]], path: str | Path) -> None:
    output_path = _prepare_path(path)
    damage = [row["damage_level"] for row in rows]
    holdout = [100.0 * row.get("holdout_phoneme_best_guess", 0.0) for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(damage, holdout, marker="o")
    plt.xlabel("Uniform weight perturbation scale")
    plt.ylabel("Best-guess phoneme accuracy (%)")
    plt.title("Controlled damage trajectory under scaled perturbations")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_relearning(
    relearn_history: list[dict[str, float]],
    fresh_history: list[dict[str, float]],
    path: str | Path,
) -> None:
    output_path = _prepare_path(path)
    relearn_epoch = [row["epoch"] for row in relearn_history]
    relearn_acc = [100.0 * row.get("holdout_phoneme_best_guess", 0.0) for row in relearn_history]
    fresh_epoch = [row["epoch"] for row in fresh_history]
    fresh_acc = [100.0 * row.get("holdout_phoneme_best_guess", 0.0) for row in fresh_history]

    plt.figure(figsize=(8, 5))
    plt.plot(relearn_epoch, relearn_acc, marker="o", label="Relearning after damage")
    plt.plot(fresh_epoch, fresh_acc, marker="o", label="Fresh training")
    plt.xlabel("Epoch")
    plt.ylabel("Holdout best-guess phoneme accuracy (%)")
    plt.title("Relearning speed comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_spacing(spacing: dict[str, float | dict[str, float]], path: str | Path) -> None:
    output_path = _prepare_path(path)
    massed = spacing.get("massed_novel_metrics", {})
    interleaved = spacing.get("interleaved_novel_metrics", {})

    metrics = [
        "novel_phoneme_best_guess",
        "novel_stress_best_guess",
        "novel_perfect_match_phoneme",
        "novel_perfect_match_all",
    ]
    labels = ["Phoneme", "Stress", "Perfect(phoneme)", "Perfect(all)"]
    massed_values = [100.0 * float(massed.get(metric, 0.0)) for metric in metrics]
    interleaved_values = [100.0 * float(interleaved.get(metric, 0.0)) for metric in metrics]

    x_positions = range(len(metrics))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar([x - width / 2 for x in x_positions], massed_values, width=width, label="Massed")
    plt.bar([x + width / 2 for x in x_positions], interleaved_values, width=width, label="Interleaved")
    plt.xticks(list(x_positions), labels)
    plt.ylabel("Accuracy (%)")
    plt.title("Spacing schedule comparison on novel words")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_power_law(rows: list[dict[str, float]], fit: dict[str, float], path: str | Path) -> None:
    output_path = _prepare_path(path)
    epochs = [row["epoch"] for row in rows if row.get("epoch", 0.0) > 0 and row.get("train_word_mse", 0.0) > 0]
    mse_values = [row["train_word_mse"] for row in rows if row.get("epoch", 0.0) > 0 and row.get("train_word_mse", 0.0) > 0]

    if not epochs:
        plt.figure(figsize=(7, 5))
        plt.title("Power-law fit unavailable")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return

    x = np.array(epochs, dtype=np.float64)
    y = np.array(mse_values, dtype=np.float64)
    intercept = float(fit.get("intercept", 0.0))
    exponent = float(fit.get("exponent", 0.0))
    y_fit = np.exp(intercept) * np.power(x, exponent)

    plt.figure(figsize=(8, 5))
    plt.loglog(x, y, marker="o", linestyle="", label="Observed")
    plt.loglog(x, y_fit, linestyle="-", label=f"Fit slope={exponent:.3f}")
    plt.xlabel("Epoch (log)")
    plt.ylabel("Train word MSE (log)")
    plt.title("Power-law learning curve fit")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_time_travel(rows: list[dict[str, float | str]], path: str | Path) -> None:
    output_path = _prepare_path(path)
    labels = [str(row["experiment"]) for row in rows]
    values = [100.0 * float(row.get("holdout_phoneme_best_guess", 0.0)) for row in rows]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.ylabel("Holdout best-guess phoneme accuracy (%)")
    plt.title("Time-travel ablations (dictionary-first)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_quantization(rows: list[dict[str, float]], path: str | Path) -> None:
    output_path = _prepare_path(path)
    rows_sorted = sorted(rows, key=lambda item: item["bits"])
    bits = [row["bits"] for row in rows_sorted]
    holdout = [100.0 * row.get("holdout_phoneme_best_guess", 0.0) for row in rows_sorted]

    plt.figure(figsize=(8, 5))
    plt.plot(bits, holdout, marker="o")
    plt.xlabel("Quantization bits per parameter")
    plt.ylabel("Holdout best-guess phoneme accuracy (%)")
    plt.title("Quantization tolerance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_hidden_cluster_heatmap(labels: list[str], distance_matrix: list[list[float]], path: str | Path) -> None:
    output_path = _prepare_path(path)
    if not labels or not distance_matrix:
        plt.figure(figsize=(6, 4))
        plt.title("Hidden representation clustering (no data)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return

    plt.figure(figsize=(12, 10))
    matrix = distance_matrix
    image = plt.imshow(matrix, cmap="viridis", interpolation="nearest")
    plt.colorbar(image, fraction=0.046, pad=0.04, label="Complete-linkage input distance")
    ticks = list(range(len(labels)))
    if len(labels) > 60:
        step = max(1, len(labels) // 20)
        tick_positions = ticks[::step]
        tick_labels = [labels[idx] for idx in tick_positions]
    else:
        tick_positions = ticks
        tick_labels = labels
    plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=90, fontsize=7)
    plt.yticks(ticks=tick_positions, labels=tick_labels, fontsize=7)
    plt.title("Hidden-unit prototype distances (cluster ordered)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_cluster_dendrogram(
    labels: list[str],
    leaf_order: list[int],
    merge_rows: list[dict[str, float]],
    path: str | Path,
    *,
    label_colors: list[str] | None = None,
) -> None:
    output_path = _prepare_path(path)
    if not labels or not merge_rows or not leaf_order:
        plt.figure(figsize=(8, 5))
        plt.title("Clustering dendrogram unavailable")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return

    n = len(labels)
    if len(leaf_order) != n:
        plt.figure(figsize=(8, 5))
        plt.title("Clustering dendrogram (invalid leaf order)")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return

    x_positions: dict[int, float] = {int(original_index): float(position) for position, original_index in enumerate(leaf_order)}
    node_x: dict[int, float] = {leaf_id: x_positions[leaf_id] for leaf_id in range(n)}
    node_y: dict[int, float] = {leaf_id: 0.0 for leaf_id in range(n)}

    plt.figure(figsize=(max(12, n * 0.22), 6))
    for row in merge_rows:
        left = int(row["left_id"])
        right = int(row["right_id"])
        new_id = int(row["new_id"])
        dist = float(row["distance"])
        left_x = node_x[left]
        right_x = node_x[right]
        left_y = node_y[left]
        right_y = node_y[right]

        plt.plot([left_x, left_x], [left_y, dist], color="black", linewidth=0.7)
        plt.plot([right_x, right_x], [right_y, dist], color="black", linewidth=0.7)
        plt.plot([left_x, right_x], [dist, dist], color="black", linewidth=0.7)
        node_x[new_id] = 0.5 * (left_x + right_x)
        node_y[new_id] = dist

    ordered_labels = labels
    tick_positions = list(range(n))
    plt.xticks(ticks=tick_positions, labels=ordered_labels, rotation=90, fontsize=6)
    if label_colors is not None and len(label_colors) == len(ordered_labels):
        axis = plt.gca()
        for tick, color in zip(axis.get_xticklabels(), label_colors):
            tick.set_color(color)
    plt.ylabel("Complete-linkage distance")
    plt.title("Hierarchical clustering of letter-to-sound correspondences (complete linkage)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_hidden_merge_distances(merge_rows: list[dict[str, float]], path: str | Path) -> None:
    output_path = _prepare_path(path)
    if not merge_rows:
        plt.figure(figsize=(7, 5))
        plt.title("No clustering merges")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return

    indices = list(range(1, len(merge_rows) + 1))
    distances = [row.get("distance", 0.0) for row in merge_rows]
    sizes = [row.get("merged_size", 0.0) for row in merge_rows]

    fig, axis_left = plt.subplots(figsize=(8, 5))
    axis_left.plot(indices, distances, marker="o", label="Merge distance")
    axis_left.set_xlabel("Merge step")
    axis_left.set_ylabel("Complete-linkage distance")
    axis_left.grid(alpha=0.3)

    axis_right = axis_left.twinx()
    axis_right.plot(indices, sizes, color="tab:orange", linestyle="--", marker="x", label="Merged size")
    axis_right.set_ylabel("Cluster size")

    lines = axis_left.get_lines() + axis_right.get_lines()
    labels = [line.get_label() for line in lines]
    axis_left.legend(lines, labels, loc="best")
    plt.title("Hidden clustering merge trajectory")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_paper_vs_ours(rows: list[dict[str, float | str]], path: str | Path) -> None:
    output_path = _prepare_path(path)
    if not rows:
        plt.figure(figsize=(8, 5))
        plt.title("Paper comparison unavailable")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return

    labels = [str(row["metric"]) for row in rows]
    paper_values = [float(row["paper_target"]) for row in rows]
    ours_values = [float(row["ours"]) for row in rows]
    x_positions = np.arange(len(rows))
    width = 0.35

    plt.figure(figsize=(11, 5))
    plt.bar(x_positions - width / 2, paper_values, width=width, label="Paper target")
    plt.bar(x_positions + width / 2, ours_values, width=width, label="Ours")
    plt.xticks(x_positions, labels, rotation=20, ha="right")
    plt.ylabel("Best-guess phoneme accuracy (%)")
    plt.title("Paper targets vs reproduced results")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
