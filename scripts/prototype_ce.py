"""Prototype cross-entropy experiments for the NETtalk single-hidden-layer MLP.

Reproduces the full decoder-aware training study end to end. Instead of per-bit MSE,
the network is trained so the paper's nearest-angle "best guess" decoder is correct:
cosine similarity between the sigmoid output block and each fixed phoneme/stress
prototype code, temperature-scaled, softmax, cross-entropy. The loss and its gradient
live in nettalk/prototype.py (finite-difference checked by scripts/check_prototype_gradient.py).

Everything is held identical to the MSE baselines for an apples-to-apples comparison:
80 hidden units, window 7, train_1000_w7 -> holdout_dict_w7, and a seed-0 900/100
by-word validation split used for ALL hyperparameter selection (the dictionary holdout
is never used to select anything).

Stages (default: all):
  grid        pure cosine-CE tau/lr/optimizer grid on the val split
                -> results/prototype_ce_grid_search.csv
  diagnosis   per-epoch holdout/val best-guess for the selected pure-CE config
                -> results/prototype_ce_peak_diagnosis.csv
  pure-final  pure cosine-CE 3-seed final on holdout
                -> results/prototype_ce_seed_sweep.csv
  hybrid      L = cosine_CE + lambda*MSE grid + 3-seed final
                -> results/prototype_ce_hybrid_grid.csv, results/prototype_ce_hybrid_final.csv
  finetune    MSE pretrain -> pure cosine-CE finetune grid + 3-seed final
                -> results/prototype_ce_finetune_grid.csv, results/prototype_ce_finetune_final.csv

Run: python scripts/prototype_ce.py [--stage all|grid|diagnosis|pure-final|hybrid|finetune]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nettalk.data import WordDataset, build_repro_artifacts, load_feature_mapping, load_preprocessed_dataset
from nettalk.eval import evaluate_dataset
from nettalk.model import NettalkMLP
from nettalk.prototype import PrototypeTrainingConfig, train_prototype_ce
from nettalk.train import paper_1987_config, train_model

# Fixed experimental setup, identical to the MSE baselines.
HIDDEN_DIM = 80
FINAL_EPOCHS = 55
GRID_EPOCHS = 20
SEEDS = (42, 43, 44)
SPLIT_SEED = 0
VAL_FRACTION = 0.1

# Selected pure cosine-CE config (validation-only selection over the grid below).
PURE_OPTIMIZER = "adam"
PURE_TAU = 0.05
PURE_LR = 0.003
PURE_SGD_TAUS = (0.05, 0.1, 0.2, 0.5)
PURE_SGD_LRS = (0.5, 1.0, 2.0)
PURE_ADAM_TAUS = (0.05, 0.1, 0.2, 0.5)
PURE_ADAM_LRS = (0.003, 0.01)

# Selected hybrid config, L = cosine_CE + lambda * MSE (validation-only selection).
HYBRID_LAMBDA = 0.3
HYBRID_TAU = 0.05
HYBRID_LR = 0.003
HYBRID_LAMBDAS = (0.1, 0.3, 1.0, 3.0)
HYBRID_TAUS = (0.05, 0.1)
HYBRID_LRS = (0.003, 0.01)

# Selected MSE-pretrain -> cosine-CE-finetune config (validation-only selection).
FINETUNE_PRETRAIN = "margin"  # paper_1987 MSE (margin gate on)
FINETUNE_FT_EPOCHS = 5
FINETUNE_FT_LR = 0.001
FINETUNE_TAU = 0.05  # pure cosine-CE temperature for the finetune stage
FINETUNE_PRETRAIN_TYPES = ("margin", "no_margin")
FINETUNE_FT_EPOCH_GRID = (5, 10, 20)
FINETUNE_FT_LRS = (0.001, 0.003)


def _write_csv(path: Path, rows: list[dict]) -> None:
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


def _ensure_preprocessed(repo_root: Path) -> None:
    repro_dir = repo_root / "data" / "processed" / "repro"
    required = (repro_dir / "train_1000_w7.npz", repro_dir / "holdout_dict_w7.npz")
    if not all(path.exists() for path in required):
        build_repro_artifacts(repo_root, window_sizes=(7,))


def split_by_word(dataset: WordDataset, *, val_fraction: float, seed: int) -> tuple[WordDataset, WordDataset]:
    """Split a word dataset into (train, val) by whole words, seedable."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset.words))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(indices) * val_fraction)))
    val_indices = sorted(indices[:n_val].tolist())
    train_indices = sorted(indices[n_val:].tolist())
    return dataset.subset(train_indices, name="gridtrain"), dataset.subset(val_indices, name="gridval")


def _summarize(rows: list[dict], key: str) -> tuple[float, float]:
    values = np.array([row[key] for row in rows], dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=0))


def _mse_config(pretrain_type: str, seed: int):
    config = paper_1987_config(epochs=FINAL_EPOCHS, seed=seed)
    if pretrain_type == "no_margin":
        config.use_margin_gate = False
    return config


def _val_selected_early_stop_epoch(
    grid_train: WordDataset,
    grid_val: WordDataset,
    mapping,
    config: PrototypeTrainingConfig,
    seed: int = 42,
) -> int:
    """Pick an early-stop epoch honestly: train seed-42 on grid_train (900) while
    monitoring grid_val (100, genuinely held out here), return the argmax epoch."""
    model = NettalkMLP(input_dim=grid_train.input_dim, hidden_dim=HIDDEN_DIM, seed=seed)
    cfg = replace(config, epochs=FINAL_EPOCHS, seed=seed)

    def callback(trained_model: NettalkMLP) -> dict[str, float]:
        return {"val_bg": evaluate_dataset(trained_model, grid_val, mapping, prefix="")["phoneme_best_guess"]}

    history = train_prototype_ce(model, grid_train, mapping, cfg, eval_callback=callback)
    val_curve = np.array([row["val_bg"] for row in history])
    return int(np.argmax(val_curve)) + 1


def _snapshot_callback(target_epoch: int):
    state: dict = {"epoch": 0, "snapshot": None}

    def callback(trained_model: NettalkMLP) -> dict[str, float]:
        state["epoch"] += 1
        if state["epoch"] == target_epoch:
            state["snapshot"] = trained_model.copy()
        return {}

    return callback, state


def _prototype_3seed_final(
    train_full: WordDataset,
    holdout: WordDataset,
    mapping,
    config: PrototypeTrainingConfig,
    early_stop_epoch: int,
) -> list[dict]:
    """3-seed full-1000 final; reports holdout at epoch 55 and at the early-stop snapshot."""
    rows: list[dict] = []
    for seed in SEEDS:
        model = NettalkMLP(input_dim=train_full.input_dim, hidden_dim=HIDDEN_DIM, seed=seed)
        cfg = replace(config, epochs=FINAL_EPOCHS, seed=seed)
        callback, state = _snapshot_callback(early_stop_epoch)
        train_prototype_ce(model, train_full, mapping, cfg, eval_callback=callback)
        final_metrics = evaluate_dataset(model, holdout, mapping, prefix="holdout_")
        snapshot = state["snapshot"] if state["snapshot"] is not None else model
        early_metrics = evaluate_dataset(snapshot, holdout, mapping, prefix="holdout_")
        rows.append({
            "seed": seed,
            "final55_phoneme_best_guess": final_metrics["holdout_phoneme_best_guess"],
            "final55_perfect_match_all": final_metrics["holdout_perfect_match_all"],
            "final55_stress_best_guess": final_metrics["holdout_stress_best_guess"],
            "earlystop_epoch": early_stop_epoch,
            "earlystop_phoneme_best_guess": early_metrics["holdout_phoneme_best_guess"],
            "earlystop_perfect_match_all": early_metrics["holdout_perfect_match_all"],
            "earlystop_stress_best_guess": early_metrics["holdout_stress_best_guess"],
        })
    return rows


def run_grid(grid_train: WordDataset, grid_val: WordDataset, mapping, results_dir: Path) -> None:
    """Pure cosine-CE tau/lr/optimizer grid, selected on validation phoneme best-guess."""
    print("== Pure cosine-CE grid (20 epochs, select on val) ==")
    rows: list[dict] = []

    def run_cell(optimizer: str, tau: float, lr: float) -> dict:
        start = time.time()
        model = NettalkMLP(input_dim=grid_train.input_dim, hidden_dim=HIDDEN_DIM, seed=42)
        config = PrototypeTrainingConfig(epochs=GRID_EPOCHS, tau=tau, learning_rate=lr, optimizer=optimizer, seed=42)
        train_prototype_ce(model, grid_train, mapping, config)
        metrics = evaluate_dataset(model, grid_val, mapping, prefix="val_")
        row = {"optimizer": optimizer, "tau": tau, "learning_rate": lr,
               "epochs": float(GRID_EPOCHS), "seconds": round(time.time() - start, 2)}
        row.update(metrics)
        print(f"  {optimizer:4s} tau={tau:<5} lr={lr:<6} val_bg={100.0 * metrics['val_phoneme_best_guess']:.2f}%")
        return row

    for tau in PURE_SGD_TAUS:
        for lr in PURE_SGD_LRS:
            rows.append(run_cell("sgd", tau, lr))
    for tau in PURE_ADAM_TAUS:
        for lr in PURE_ADAM_LRS:
            rows.append(run_cell("adam", tau, lr))

    best = max(rows, key=lambda r: r["val_phoneme_best_guess"])
    print(f"  selected: {best['optimizer']} tau={best['tau']} lr={best['learning_rate']} "
          f"val_bg={100.0 * best['val_phoneme_best_guess']:.2f}%")
    _write_csv(results_dir / "prototype_ce_grid_search.csv", rows)


def run_diagnosis(train_full: WordDataset, holdout: WordDataset, grid_val: WordDataset, mapping, results_dir: Path) -> None:
    """Per-epoch holdout/val best-guess curve for the selected pure-CE config (seed 42)."""
    print("== Pure cosine-CE epoch diagnosis (seed 42) ==")
    model = NettalkMLP(input_dim=train_full.input_dim, hidden_dim=HIDDEN_DIM, seed=42)
    config = PrototypeTrainingConfig(epochs=FINAL_EPOCHS, tau=PURE_TAU, learning_rate=PURE_LR, optimizer=PURE_OPTIMIZER, seed=42)

    def callback(trained_model: NettalkMLP) -> dict[str, float]:
        return {
            "holdout_phoneme_best_guess": evaluate_dataset(trained_model, holdout, mapping, prefix="")["phoneme_best_guess"],
            "val_phoneme_best_guess": evaluate_dataset(trained_model, grid_val, mapping, prefix="")["phoneme_best_guess"],
        }

    history = train_prototype_ce(model, train_full, mapping, config, eval_callback=callback)
    rows = [{"epoch": int(row["epoch"]),
             "holdout_phoneme_best_guess": row["holdout_phoneme_best_guess"],
             "val_phoneme_best_guess": row["val_phoneme_best_guess"]} for row in history]

    holdout_curve = np.array([row["holdout_phoneme_best_guess"] for row in rows])
    peak_idx = int(np.argmax(holdout_curve))
    print(f"  peak holdout best-guess {100.0 * holdout_curve[peak_idx]:.2f}% at epoch {peak_idx + 1}; "
          f"epoch-55 {100.0 * holdout_curve[-1]:.2f}%")
    _write_csv(results_dir / "prototype_ce_peak_diagnosis.csv", rows)


def run_pure_final(train_full: WordDataset, holdout: WordDataset, mapping, results_dir: Path) -> None:
    """Pure cosine-CE 3-seed final on the dictionary holdout."""
    print("== Pure cosine-CE 3-seed final ==")
    rows: list[dict] = []
    for seed in SEEDS:
        start = time.time()
        model = NettalkMLP(input_dim=train_full.input_dim, hidden_dim=HIDDEN_DIM, seed=seed)
        config = PrototypeTrainingConfig(epochs=FINAL_EPOCHS, tau=PURE_TAU, learning_rate=PURE_LR, optimizer=PURE_OPTIMIZER, seed=seed)
        train_prototype_ce(model, train_full, mapping, config)
        train_seconds = time.time() - start
        eval_start = time.time()
        metrics = evaluate_dataset(model, holdout, mapping, prefix="holdout_")
        row = {"seed": seed, "optimizer": PURE_OPTIMIZER, "tau": PURE_TAU, "learning_rate": PURE_LR,
               "hidden_dim": HIDDEN_DIM, "epochs": FINAL_EPOCHS,
               "train_seconds": round(train_seconds, 2), "eval_seconds": round(time.time() - eval_start, 2)}
        row.update(metrics)
        rows.append(row)
        print(f"  seed={seed} holdout_bg={100.0 * metrics['holdout_phoneme_best_guess']:.2f}% "
              f"perfect_all={100.0 * metrics['holdout_perfect_match_all']:.2f}%")
    bg_mean, bg_std = _summarize(rows, "holdout_phoneme_best_guess")
    print(f"  holdout best-guess {100.0 * bg_mean:.2f}% +/- {100.0 * bg_std:.2f}%")
    _write_csv(results_dir / "prototype_ce_seed_sweep.csv", rows)


def run_hybrid(grid_train, grid_val, train_full, holdout, mapping, results_dir: Path) -> None:
    """Hybrid L = cosine_CE + lambda*MSE: grid on val, then 3-seed final on holdout."""
    print("== Hybrid cosine-CE + lambda*MSE grid (20 epochs, select on val) ==")
    grid_rows: list[dict] = []
    for lam in HYBRID_LAMBDAS:
        for tau in HYBRID_TAUS:
            for lr in HYBRID_LRS:
                model = NettalkMLP(input_dim=grid_train.input_dim, hidden_dim=HIDDEN_DIM, seed=42)
                config = PrototypeTrainingConfig(epochs=GRID_EPOCHS, tau=tau, mse_weight=lam, learning_rate=lr, optimizer="adam", seed=42)
                train_prototype_ce(model, grid_train, mapping, config)
                metrics = evaluate_dataset(model, grid_val, mapping, prefix="val_")
                grid_rows.append({"mse_weight": lam, "tau": tau, "learning_rate": lr,
                                  "val_phoneme_best_guess": metrics["val_phoneme_best_guess"],
                                  "val_perfect_match_all": metrics["val_perfect_match_all"]})
    best = max(grid_rows, key=lambda r: r["val_phoneme_best_guess"])
    print(f"  grid winner: lambda={best['mse_weight']} tau={best['tau']} lr={best['learning_rate']} "
          f"val_bg={100.0 * best['val_phoneme_best_guess']:.2f}% (selected constant lambda={HYBRID_LAMBDA})")
    _write_csv(results_dir / "prototype_ce_hybrid_grid.csv", grid_rows)

    config = PrototypeTrainingConfig(tau=HYBRID_TAU, mse_weight=HYBRID_LAMBDA, learning_rate=HYBRID_LR, optimizer="adam")
    early_stop_epoch = _val_selected_early_stop_epoch(grid_train, grid_val, mapping, config)
    print(f"  val-selected early-stop epoch E*={early_stop_epoch}")
    final_rows = _prototype_3seed_final(train_full, holdout, mapping, config, early_stop_epoch)
    bg_mean, bg_std = _summarize(final_rows, "earlystop_phoneme_best_guess")
    print(f"  holdout best-guess (early-stop) {100.0 * bg_mean:.2f}% +/- {100.0 * bg_std:.2f}%")
    _write_csv(results_dir / "prototype_ce_hybrid_final.csv", final_rows)


def run_finetune(grid_train, grid_val, train_full, holdout, mapping, results_dir: Path) -> None:
    """MSE pretrain -> pure cosine-CE finetune: grid on val, then 3-seed final on holdout."""
    print("== MSE pretrain -> cosine-CE finetune grid (select on val) ==")
    pretrained = {}
    for pretrain_type in FINETUNE_PRETRAIN_TYPES:
        model = NettalkMLP(input_dim=grid_train.input_dim, hidden_dim=HIDDEN_DIM, seed=42)
        train_model(model, grid_train, _mse_config(pretrain_type, 42))
        pretrained[pretrain_type] = model

    grid_rows: list[dict] = []
    for pretrain_type in FINETUNE_PRETRAIN_TYPES:
        for ft_epochs in FINETUNE_FT_EPOCH_GRID:
            for ft_lr in FINETUNE_FT_LRS:
                model = pretrained[pretrain_type].copy()
                config = PrototypeTrainingConfig(epochs=ft_epochs, tau=FINETUNE_TAU, learning_rate=ft_lr, optimizer="adam", seed=42)
                train_prototype_ce(model, grid_train, mapping, config)
                metrics = evaluate_dataset(model, grid_val, mapping, prefix="val_")
                grid_rows.append({"pretrain_type": pretrain_type, "ft_epochs": ft_epochs, "ft_lr": ft_lr,
                                  "val_phoneme_best_guess": metrics["val_phoneme_best_guess"],
                                  "val_perfect_match_all": metrics["val_perfect_match_all"]})
    best = max(grid_rows, key=lambda r: r["val_phoneme_best_guess"])
    print(f"  grid winner: pretrain={best['pretrain_type']} ft_epochs={best['ft_epochs']} ft_lr={best['ft_lr']} "
          f"val_bg={100.0 * best['val_phoneme_best_guess']:.2f}% "
          f"(selected constant {FINETUNE_PRETRAIN}/{FINETUNE_FT_EPOCHS}ep/lr={FINETUNE_FT_LR})")
    _write_csv(results_dir / "prototype_ce_finetune_grid.csv", grid_rows)

    final_rows: list[dict] = []
    for seed in SEEDS:
        model = NettalkMLP(input_dim=train_full.input_dim, hidden_dim=HIDDEN_DIM, seed=seed)
        train_model(model, train_full, _mse_config(FINETUNE_PRETRAIN, seed))
        pre = evaluate_dataset(model, holdout, mapping, prefix="holdout_")
        config = PrototypeTrainingConfig(epochs=FINETUNE_FT_EPOCHS, tau=FINETUNE_TAU, learning_rate=FINETUNE_FT_LR, optimizer="adam", seed=seed)
        train_prototype_ce(model, train_full, mapping, config)
        post = evaluate_dataset(model, holdout, mapping, prefix="holdout_")
        final_rows.append({"seed": seed,
                           "pre_finetune_phoneme_best_guess": pre["holdout_phoneme_best_guess"],
                           "phoneme_best_guess": post["holdout_phoneme_best_guess"],
                           "perfect_match_all": post["holdout_perfect_match_all"],
                           "stress_best_guess": post["holdout_stress_best_guess"]})
    bg_mean, bg_std = _summarize(final_rows, "phoneme_best_guess")
    pre_mean, _ = _summarize(final_rows, "pre_finetune_phoneme_best_guess")
    print(f"  holdout best-guess {100.0 * bg_mean:.2f}% +/- {100.0 * bg_std:.2f}% "
          f"(pre-finetune MSE {100.0 * pre_mean:.2f}%)")
    _write_csv(results_dir / "prototype_ce_finetune_final.csv", final_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype cross-entropy experiments for NETtalk.")
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="Repository root path.")
    parser.add_argument(
        "--stage",
        choices=("all", "grid", "diagnosis", "pure-final", "hybrid", "finetune"),
        default="all",
        help="Which stage to run.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    _ensure_preprocessed(repo_root)
    mapping = load_feature_mapping(repo_root)
    repro_dir = repo_root / "data" / "processed" / "repro"
    train_full = load_preprocessed_dataset(repro_dir / "train_1000_w7.npz")
    holdout = load_preprocessed_dataset(repro_dir / "holdout_dict_w7.npz")
    grid_train, grid_val = split_by_word(train_full, val_fraction=VAL_FRACTION, seed=SPLIT_SEED)

    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    if args.stage in ("all", "grid"):
        run_grid(grid_train, grid_val, mapping, results_dir)
    if args.stage in ("all", "diagnosis"):
        run_diagnosis(train_full, holdout, grid_val, mapping, results_dir)
    if args.stage in ("all", "pure-final"):
        run_pure_final(train_full, holdout, mapping, results_dir)
    if args.stage in ("all", "hybrid"):
        run_hybrid(grid_train, grid_val, train_full, holdout, mapping, results_dir)
    if args.stage in ("all", "finetune"):
        run_finetune(grid_train, grid_val, train_full, holdout, mapping, results_dir)

    print(f"Done ({args.stage}) in {time.time() - start:.1f}s.")


if __name__ == "__main__":
    main()
