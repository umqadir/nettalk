"""Run controlled modernization ("time-travel") ablations for NETtalk."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from nettalk.data import FeatureMapping, WordDataset, build_repro_artifacts, load_feature_mapping, load_preprocessed_dataset
from nettalk.eval import evaluate_dataset
from nettalk.model import NettalkMLP
from nettalk.plots import plot_time_travel
from nettalk.prototype import PrototypeTrainingConfig, train_prototype_ce
from nettalk.train import TrainingConfig, paper_1987_config, train_model

# Selected by scripts/prototype_ce_grid_search.py (validation-only, no holdout leakage;
# see results/prototype_ce_grid_search.csv for the full selection trace).
PROTOTYPE_CE_OPTIMIZER = "adam"
PROTOTYPE_CE_TAU = 0.05
PROTOTYPE_CE_LR = 0.003
PROTOTYPE_CE_SEED = 42


def _write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _ensure_preprocessed(repo_root: Path) -> None:
    repro_dir = repo_root / "data" / "processed" / "repro"
    manifest_path = repro_dir / "manifest.json"
    required_windows = (7, 11)
    needs_rebuild = not manifest_path.exists()
    if not needs_rebuild:
        for window in required_windows:
            if not (repro_dir / f"train_1000_w{window}.npz").exists():
                needs_rebuild = True
                break
            if not (repro_dir / f"holdout_dict_w{window}.npz").exists():
                needs_rebuild = True
                break
    if needs_rebuild:
        build_repro_artifacts(repo_root, window_sizes=(3, 5, 7, 9, 11))


def _run_single_experiment(
    name: str,
    train_dataset: WordDataset,
    holdout_dataset: WordDataset,
    mapping: FeatureMapping,
    config: TrainingConfig,
    hidden_dim: int,
    seed: int,
) -> dict[str, float | str]:
    model = NettalkMLP(input_dim=train_dataset.input_dim, hidden_dim=hidden_dim, seed=seed)
    train_model(model, train_dataset, config)
    metrics = evaluate_dataset(model, holdout_dataset, mapping, prefix="holdout_")
    row: dict[str, float | str] = {"experiment": name, "hidden_dim": float(hidden_dim)}
    row.update(metrics)
    return row


def _run_single_prototype_experiment(
    name: str,
    train_dataset: WordDataset,
    holdout_dataset: WordDataset,
    mapping: FeatureMapping,
    config: PrototypeTrainingConfig,
    hidden_dim: int,
    seed: int,
) -> dict[str, float | str]:
    model = NettalkMLP(input_dim=train_dataset.input_dim, hidden_dim=hidden_dim, seed=seed)
    train_prototype_ce(model, train_dataset, mapping, config)
    metrics = evaluate_dataset(model, holdout_dataset, mapping, prefix="holdout_")
    row: dict[str, float | str] = {"experiment": name, "hidden_dim": float(hidden_dim)}
    row.update(metrics)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NETtalk time-travel ablations.")
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="Repository root path.")
    parser.add_argument(
        "--profile",
        choices=("paper", "fast"),
        default="paper",
        help="Execution profile. 'paper' uses pass counts closer to published runs.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs per ablation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    if args.epochs is not None:
        ablation_epochs = args.epochs
    elif args.profile == "paper":
        ablation_epochs = 55
    else:
        ablation_epochs = 8

    repo_root = args.repo_root.resolve()
    _ensure_preprocessed(repo_root)

    mapping = load_feature_mapping(repo_root)
    repro_dir = repo_root / "data" / "processed" / "repro"
    train_w7 = load_preprocessed_dataset(repro_dir / "train_1000_w7.npz")
    holdout_w7 = load_preprocessed_dataset(repro_dir / "holdout_dict_w7.npz")
    train_w11 = load_preprocessed_dataset(repro_dir / "train_1000_w11.npz")
    holdout_w11 = load_preprocessed_dataset(repro_dir / "holdout_dict_w11.npz")

    baseline_config = paper_1987_config(epochs=ablation_epochs, seed=args.seed)
    no_margin_config = paper_1987_config(epochs=ablation_epochs, seed=args.seed + 1)
    no_margin_config.use_margin_gate = False

    adam_config = paper_1987_config(epochs=ablation_epochs, seed=args.seed + 2)
    adam_config.optimizer = "adam"
    adam_config.learning_rate = 0.01
    adam_config.use_margin_gate = False

    prototype_ce_config = PrototypeTrainingConfig(
        epochs=ablation_epochs,
        tau=PROTOTYPE_CE_TAU,
        stress_weight=1.0,
        learning_rate=PROTOTYPE_CE_LR,
        optimizer=PROTOTYPE_CE_OPTIMIZER,
        seed=PROTOTYPE_CE_SEED,
    )

    modern_rows = [
        _run_single_experiment(
            "baseline_1987_w7",
            train_w7,
            holdout_w7,
            mapping,
            baseline_config,
            hidden_dim=80,
            seed=args.seed,
        ),
        _run_single_experiment(
            "no_margin_w7",
            train_w7,
            holdout_w7,
            mapping,
            no_margin_config,
            hidden_dim=80,
            seed=args.seed + 1,
        ),
        _run_single_experiment(
            "adam_no_margin_w7",
            train_w7,
            holdout_w7,
            mapping,
            adam_config,
            hidden_dim=80,
            seed=args.seed + 2,
        ),
        _run_single_experiment(
            "baseline_1987_w11_param_matched",
            train_w11,
            holdout_w11,
            mapping,
            baseline_config,
            hidden_dim=56,
            seed=args.seed + 3,
        ),
        _run_single_prototype_experiment(
            "prototype_ce_w7",
            train_w7,
            holdout_w7,
            mapping,
            prototype_ce_config,
            hidden_dim=80,
            seed=PROTOTYPE_CE_SEED,
        ),
    ]

    results_dir = repo_root / "results"
    figs_dir = repo_root / "figs"
    results_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(results_dir / "time_travel_ablation.csv", modern_rows)
    plot_time_travel(modern_rows, figs_dir / "time_travel_ablation.png")
    (results_dir / "time_travel_meta.json").write_text(
        f'{{"profile":"{args.profile}","epochs":{ablation_epochs}}}',
        encoding="utf-8",
    )

    print("Time-travel ablations complete.")
    print(results_dir / "time_travel_ablation.csv")


if __name__ == "__main__":
    main()
