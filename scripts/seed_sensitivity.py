"""Run targeted seed-sensitivity diagnostics for headline NETtalk results."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nettalk.data import build_repro_artifacts, load_feature_mapping, load_preprocessed_dataset
from nettalk.experiments import run_generalization_experiment, run_two_layer_network_experiment


def _ensure_preprocessed(repo_root: Path) -> None:
    repro_dir = repo_root / "data" / "processed" / "repro"
    required = (
        repro_dir / "train_1000_w7.npz",
        repro_dir / "holdout_dict_w7.npz",
        repro_dir / "full_dict_w7.npz",
    )
    if not all(path.exists() for path in required):
        build_repro_artifacts(repo_root, window_sizes=(7,))


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_seed_sensitivity(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    architecture_labels = {
        "one_layer_h120": "One hidden layer (H=120)",
        "two_layer_80_80": "Two hidden layers (80,80)",
    }
    paper_targets = {
        "one_layer_h120": {0.0: 77.0, 1.0: 85.0, 5.0: 90.0},
        "two_layer_80_80": {0.0: 80.0, 1.0: 87.0, 11.0: 91.0},
    }

    for axis, architecture in zip(axes, architecture_labels):
        architecture_rows = [row for row in rows if row["architecture"] == architecture]
        seeds = sorted({int(row["seed"]) for row in architecture_rows})
        for seed in seeds:
            seed_rows = sorted(
                [row for row in architecture_rows if int(row["seed"]) == seed],
                key=lambda item: float(item["dictionary_pass"]),
            )
            passes = [float(row["dictionary_pass"]) for row in seed_rows]
            accuracy = [100.0 * float(row["full_phoneme_best_guess"]) for row in seed_rows]
            axis.plot(passes, accuracy, marker="o", alpha=0.7, label=f"seed {seed}")

        for target_pass, target_value in paper_targets[architecture].items():
            axis.scatter(
                [target_pass],
                [target_value],
                marker="x",
                color="black",
                s=70,
                linewidths=1.8,
                zorder=5,
            )

        axis.set_title(architecture_labels[architecture])
        axis.set_xlabel("Dictionary passes after 1000-word pretraining")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    axes[0].set_ylabel("Full dictionary best-guess phoneme accuracy (%)")
    fig.suptitle("Seed sensitivity for dictionary generalization")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _summary(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    summary_rows: list[dict[str, float | str]] = []
    architectures = sorted({str(row["architecture"]) for row in rows})
    for architecture in architectures:
        architecture_rows = [row for row in rows if row["architecture"] == architecture]
        passes = sorted({float(row["dictionary_pass"]) for row in architecture_rows})
        for dictionary_pass in passes:
            values = np.array(
                [
                    float(row["full_phoneme_best_guess"])
                    for row in architecture_rows
                    if float(row["dictionary_pass"]) == dictionary_pass
                ],
                dtype=np.float64,
            )
            summary_rows.append(
                {
                    "architecture": architecture,
                    "dictionary_pass": dictionary_pass,
                    "n_seeds": float(len(values)),
                    "full_phoneme_best_guess_mean": float(np.mean(values)),
                    "full_phoneme_best_guess_min": float(np.min(values)),
                    "full_phoneme_best_guess_max": float(np.max(values)),
                    "full_phoneme_best_guess_std": float(np.std(values, ddof=0)),
                }
            )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run seed sensitivity for NETtalk generalization experiments.")
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Seeds to evaluate.")
    parser.add_argument("--pretrain-epochs", type=int, default=55, help="1000-word pretraining epochs.")
    parser.add_argument("--one-layer-full-passes", type=int, default=5, help="Full-dictionary passes for H=120.")
    parser.add_argument("--two-layer-full-passes", type=int, default=11, help="Full-dictionary passes for 80,80.")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    _ensure_preprocessed(repo_root)
    mapping = load_feature_mapping(repo_root)
    repro_dir = repo_root / "data" / "processed" / "repro"
    train_dataset = load_preprocessed_dataset(repro_dir / "train_1000_w7.npz")
    holdout_dataset = load_preprocessed_dataset(repro_dir / "holdout_dict_w7.npz")
    full_dataset = load_preprocessed_dataset(repro_dir / "full_dict_w7.npz")

    rows: list[dict[str, float | str]] = []
    for seed in args.seeds:
        one_layer = run_generalization_experiment(
            train_dataset,
            holdout_dataset,
            full_dataset,
            mapping,
            hidden_dim=120,
            pretrain_epochs=args.pretrain_epochs,
            full_passes=args.one_layer_full_passes,
            seed=seed,
        )
        for row in one_layer["dictionary_pass_rows"]:
            rows.append({"architecture": "one_layer_h120", "seed": float(seed), **row})

        two_layer = run_two_layer_network_experiment(
            train_dataset,
            holdout_dataset,
            full_dataset,
            mapping,
            hidden_dims=(80, 80),
            pretrain_epochs=args.pretrain_epochs,
            full_passes=args.two_layer_full_passes,
            seed=seed,
        )
        for row in two_layer["dictionary_pass_rows"]:
            rows.append({"architecture": "two_layer_80_80", "seed": float(seed), **row})

    results_dir = repo_root / "results"
    figs_dir = repo_root / "figs"
    _write_csv(results_dir / "seed_sensitivity_dictionary_generalization.csv", rows)
    _write_csv(results_dir / "seed_sensitivity_summary.csv", _summary(rows))
    _plot_seed_sensitivity(rows, figs_dir / "seed_sensitivity_dictionary_generalization.png")

    print("Seed sensitivity complete.")
    print(results_dir / "seed_sensitivity_dictionary_generalization.csv")
    print(results_dir / "seed_sensitivity_summary.csv")
    print(figs_dir / "seed_sensitivity_dictionary_generalization.png")


if __name__ == "__main__":
    main()
