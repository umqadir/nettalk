"""Run baseline NETtalk reproduction experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from nettalk.data import build_repro_artifacts, load_feature_mapping, load_preprocessed_dataset
from nettalk.experiments import (
    fit_power_law,
    run_damage_and_relearning_experiment,
    run_generalization_experiment,
    run_hidden_clustering_experiment,
    run_hidden_scaling_experiment,
    run_quantization_experiment,
    run_spacing_experiment,
    run_two_layer_network_experiment,
    run_window_size_experiment,
)
from nettalk.plots import (
    plot_cluster_dendrogram,
    plot_damage,
    plot_generalization,
    plot_generalization_comparison,
    plot_hidden_cluster_heatmap,
    plot_hidden_merge_distances,
    plot_paper_vs_ours,
    plot_power_law,
    plot_pretrain_history,
    plot_rule_learning,
    plot_scaling_histories,
    plot_hidden_scaling,
    plot_quantization,
    plot_relearning,
    plot_spacing,
    plot_window_size_learning,
)


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
    required_windows = (3, 5, 7, 9, 11)
    needs_rebuild = not manifest_path.exists()
    if not needs_rebuild:
        for window in required_windows:
            if not (repro_dir / f"train_1000_w{window}.npz").exists():
                needs_rebuild = True
                break
            if not (repro_dir / f"holdout_dict_w{window}.npz").exists():
                needs_rebuild = True
                break
            if not (repro_dir / f"full_dict_w{window}.npz").exists():
                needs_rebuild = True
                break
    if needs_rebuild:
        build_repro_artifacts(repo_root, window_sizes=required_windows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NETtalk dictionary-first reproduction experiments.")
    parser.add_argument("--repo-root", type=Path, default=Path("."), help="Repository root path.")
    parser.add_argument(
        "--profile",
        choices=("paper", "fast"),
        default="paper",
        help="Execution profile. 'paper' uses pass counts closer to published runs.",
    )
    parser.add_argument("--scaling-epochs", type=int, default=None, help="Epochs for hidden-size sweep.")
    parser.add_argument("--generalization-pretrain-epochs", type=int, default=None, help="Pretrain epochs on 1000-word subset.")
    parser.add_argument("--generalization-full-passes", type=int, default=None, help="Passes on full dictionary after pretraining.")
    parser.add_argument("--damage-pretrain-epochs", type=int, default=None, help="Pretrain epochs before damage experiment.")
    parser.add_argument("--spacing-pretrain-epochs", type=int, default=None, help="Pretrain epochs before spacing proxy.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    if args.profile == "paper":
        scaling_epochs = 55 if args.scaling_epochs is None else args.scaling_epochs
        generalization_pretrain_epochs = 55 if args.generalization_pretrain_epochs is None else args.generalization_pretrain_epochs
        generalization_full_passes = 5 if args.generalization_full_passes is None else args.generalization_full_passes
        damage_pretrain_epochs = 25 if args.damage_pretrain_epochs is None else args.damage_pretrain_epochs
        spacing_pretrain_epochs = 25 if args.spacing_pretrain_epochs is None else args.spacing_pretrain_epochs
    else:
        scaling_epochs = 8 if args.scaling_epochs is None else args.scaling_epochs
        generalization_pretrain_epochs = 8 if args.generalization_pretrain_epochs is None else args.generalization_pretrain_epochs
        generalization_full_passes = 5 if args.generalization_full_passes is None else args.generalization_full_passes
        damage_pretrain_epochs = 8 if args.damage_pretrain_epochs is None else args.damage_pretrain_epochs
        spacing_pretrain_epochs = 6 if args.spacing_pretrain_epochs is None else args.spacing_pretrain_epochs

    repo_root = args.repo_root.resolve()
    _ensure_preprocessed(repo_root)

    mapping = load_feature_mapping(repo_root)
    repro_dir = repo_root / "data" / "processed" / "repro"
    train_dataset = load_preprocessed_dataset(repro_dir / "train_1000_w7.npz")
    holdout_dataset = load_preprocessed_dataset(repro_dir / "holdout_dict_w7.npz")
    full_dataset = load_preprocessed_dataset(repro_dir / "full_dict_w7.npz")

    results_dir = repo_root / "results"
    figs_dir = repo_root / "figs"
    results_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    scaling_rows, scaling_histories = run_hidden_scaling_experiment(
        train_dataset,
        holdout_dataset,
        mapping,
        epochs=scaling_epochs,
        seed=args.seed,
    )
    _write_csv(results_dir / "scaling_summary.csv", scaling_rows)
    plot_hidden_scaling(scaling_rows, figs_dir / "scaling_hidden_units.png")
    plot_scaling_histories(scaling_histories, figs_dir / "scaling_histories.png")

    for hidden_dim, history_rows in scaling_histories.items():
        _write_csv(results_dir / f"scaling_history_hidden_{hidden_dim}.csv", history_rows)

    generalization = run_generalization_experiment(
        train_dataset,
        holdout_dataset,
        full_dataset,
        mapping,
        hidden_dim=120,
        pretrain_epochs=generalization_pretrain_epochs,
        full_passes=generalization_full_passes,
        seed=args.seed,
    )
    _write_csv(results_dir / "generalization_pretrain_history.csv", generalization["pretrain_history"])
    _write_csv(results_dir / "generalization_dictionary_passes.csv", generalization["dictionary_pass_rows"])
    plot_pretrain_history(generalization["pretrain_history"], figs_dir / "generalization_pretrain_history.png")
    plot_generalization(generalization["dictionary_pass_rows"], figs_dir / "generalization_dictionary_passes.png")
    plot_rule_learning(generalization["pretrain_history"], figs_dir / "hard_soft_c_learning.png")

    hard_soft_rows = [
        {
            "epoch": float(row["epoch"]),
            "train_hard_c_accuracy": float(row.get("train_hard_c_accuracy", 0.0)),
            "train_soft_c_accuracy": float(row.get("train_soft_c_accuracy", 0.0)),
        }
        for row in generalization["pretrain_history"]
    ]
    _write_csv(results_dir / "hard_soft_c_learning.csv", hard_soft_rows)

    # Window-size learning experiment (Figure 6 text: input groups varied 3..11).
    train_by_window = {
        window: load_preprocessed_dataset(repro_dir / f"train_1000_w{window}.npz") for window in (3, 5, 7, 9, 11)
    }
    holdout_by_window = {
        window: load_preprocessed_dataset(repro_dir / f"holdout_dict_w{window}.npz") for window in (3, 5, 7, 9, 11)
    }
    window_size_experiment = run_window_size_experiment(
        train_by_window,
        holdout_by_window,
        mapping,
        hidden_dim=80,
        epochs=scaling_epochs,
        seed=args.seed,
    )
    _write_csv(results_dir / "window_size_learning.csv", window_size_experiment["rows"])
    plot_window_size_learning(window_size_experiment["history_by_window"], figs_dir / "window_size_learning.png")

    for window_size, history in window_size_experiment["history_by_window"].items():
        _write_csv(results_dir / f"window_size_history_w{window_size}.csv", history)

    # Two-hidden-layer network experiment (1987: two layers of 80).
    two_layer = run_two_layer_network_experiment(
        train_dataset,
        holdout_dataset,
        full_dataset,
        mapping,
        hidden_dims=(80, 80),
        pretrain_epochs=generalization_pretrain_epochs,
        full_passes=11 if args.profile == "paper" else 5,
        seed=args.seed,
    )
    _write_csv(results_dir / "two_layer_pretrain_history.csv", two_layer["pretrain_history"])
    _write_csv(results_dir / "two_layer_dictionary_passes.csv", two_layer["dictionary_pass_rows"])
    plot_pretrain_history(two_layer["pretrain_history"], figs_dir / "two_layer_pretrain_history.png")
    plot_generalization(two_layer["dictionary_pass_rows"], figs_dir / "two_layer_dictionary_passes.png")
    plot_generalization_comparison(
        generalization["dictionary_pass_rows"],
        two_layer["dictionary_pass_rows"],
        figs_dir / "wide_vs_deep_generalization.png",
    )

    damage = run_damage_and_relearning_experiment(
        train_dataset,
        holdout_dataset,
        mapping,
        hidden_dim=80,
        pretrain_epochs=damage_pretrain_epochs,
        seed=args.seed,
    )
    _write_csv(results_dir / "damage_curve.csv", damage["damage_rows"])
    _write_csv(results_dir / "relearning_after_damage.csv", damage["relearn_history"])
    _write_csv(results_dir / "fresh_relearning_baseline.csv", damage["fresh_history"])
    plot_damage(damage["damage_rows"], figs_dir / "damage_curve.png")
    plot_relearning(damage["relearn_history"], damage["fresh_history"], figs_dir / "relearning_vs_fresh.png")

    spacing = run_spacing_experiment(
        train_dataset,
        holdout_dataset,
        mapping,
        hidden_dim=80,
        pretrain_epochs=spacing_pretrain_epochs,
        seed=args.seed,
    )
    spacing_path = results_dir / "spacing_experiment.json"
    spacing_path.write_text(json.dumps(spacing, indent=2), encoding="utf-8")
    plot_spacing(spacing, figs_dir / "spacing_experiment.png")

    quantization_rows = run_quantization_experiment(
        generalization["model"],
        holdout_dataset,
        mapping,
    )
    _write_csv(results_dir / "quantization_curve.csv", quantization_rows)
    plot_quantization(quantization_rows, figs_dir / "quantization_curve.png")

    clustering = run_hidden_clustering_experiment(
        generalization["model"],
        train_dataset,
    )
    _write_csv(results_dir / "hidden_clustering_merges.csv", clustering["merge_rows"])
    (results_dir / "hidden_clustering.json").write_text(json.dumps(clustering, indent=2), encoding="utf-8")
    plot_hidden_cluster_heatmap(
        clustering["labels"],
        clustering["distance_matrix_ordered"],
        figs_dir / "hidden_cluster_heatmap.png",
    )
    plot_hidden_merge_distances(clustering["merge_rows"], figs_dir / "hidden_cluster_merges.png")

    # Color correspondences in dendrogram by phoneme type (vowel vs consonant-like) using feature inventory.
    vowel_features = {"High", "Medium", "Low"}
    dendro_colors: list[str] = []
    for label in clustering["labels"]:
        try:
            phoneme_symbol = label.split("->", 1)[1]
        except IndexError:
            dendro_colors.append("black")
            continue
        features = set(mapping.phoneme_features.get(phoneme_symbol, ()))
        if features & vowel_features:
            dendro_colors.append("tab:red")
        elif phoneme_symbol in {".", "-"}:
            dendro_colors.append("tab:gray")
        else:
            dendro_colors.append("tab:blue")
    plot_cluster_dendrogram(
        clustering["labels"],
        [int(x) for x in clustering["leaf_order"]],
        clustering["merge_rows"],
        figs_dir / "hidden_cluster_dendrogram.png",
        label_colors=dendro_colors,
    )

    power_law = fit_power_law(generalization["pretrain_history"])
    (results_dir / "power_law_fit.json").write_text(json.dumps(power_law, indent=2), encoding="utf-8")
    plot_power_law(generalization["pretrain_history"], power_law, figs_dir / "power_law_fit.png")

    h0_row = next(row for row in scaling_rows if int(row["hidden_dim"]) == 0)
    h120_row = next(row for row in scaling_rows if int(row["hidden_dim"]) == 120)
    pass0_row = generalization["dictionary_pass_rows"][0]
    pass1_row = generalization["dictionary_pass_rows"][1]
    pass5_row = generalization["dictionary_pass_rows"][-1]

    # Window-size paper targets reference: w=7 H=80 -> 95%, w=11 H=80 -> 97.5% after 55 passes.
    w7_rows = window_size_experiment["history_by_window"][7]
    w11_rows = window_size_experiment["history_by_window"][11]
    w7_final_train = 100.0 * w7_rows[-1].get("train_phoneme_best_guess", 0.0)
    w11_final_train = 100.0 * w11_rows[-1].get("train_phoneme_best_guess", 0.0)

    # Two-layer paper targets reference: 97% on 1000 words after 55 passes; 80/87/91 on full dict at pass 0/1/11.
    deep_train_final = 100.0 * two_layer["pretrain_history"][-1].get("train_phoneme_best_guess", 0.0)
    deep_pass0 = two_layer["dictionary_pass_rows"][0]
    deep_pass1 = two_layer["dictionary_pass_rows"][1] if len(two_layer["dictionary_pass_rows"]) > 1 else deep_pass0
    deep_pass11 = two_layer["dictionary_pass_rows"][-1]

    paper_comparison_rows = [
        {"metric": "1000w H0 (train)", "paper_target": 82.0, "ours": 100.0 * h0_row["train_phoneme_best_guess"]},
        {"metric": "1000w H120 (train)", "paper_target": 98.0, "ours": 100.0 * h120_row["train_phoneme_best_guess"]},
        {"metric": "Gen pass0 (full)", "paper_target": 77.0, "ours": 100.0 * pass0_row["full_phoneme_best_guess"]},
        {"metric": "Gen pass1 (full)", "paper_target": 85.0, "ours": 100.0 * pass1_row["full_phoneme_best_guess"]},
        {"metric": "Gen pass5 (full)", "paper_target": 90.0, "ours": 100.0 * pass5_row["full_phoneme_best_guess"]},
        {"metric": "1000w w7 H80 (train)", "paper_target": 95.0, "ours": w7_final_train},
        {"metric": "1000w w11 H80 (train)", "paper_target": 97.5, "ours": w11_final_train},
        {"metric": "2-layer 1000w (train)", "paper_target": 97.0, "ours": deep_train_final},
        {"metric": "2-layer Gen pass0 (full)", "paper_target": 80.0, "ours": 100.0 * deep_pass0["full_phoneme_best_guess"]},
        {"metric": "2-layer Gen pass1 (full)", "paper_target": 87.0, "ours": 100.0 * deep_pass1["full_phoneme_best_guess"]},
        {"metric": "2-layer Gen pass11 (full)", "paper_target": 91.0, "ours": 100.0 * deep_pass11["full_phoneme_best_guess"]},
    ]
    _write_csv(results_dir / "paper_target_comparison.csv", paper_comparison_rows)
    plot_paper_vs_ours(paper_comparison_rows, figs_dir / "paper_target_comparison.png")

    report = {
        "profile": args.profile,
        "scaling_epochs": float(scaling_epochs),
        "generalization_pretrain_epochs": float(generalization_pretrain_epochs),
        "generalization_full_passes": float(generalization_full_passes),
        "damage_pretrain_epochs": float(damage_pretrain_epochs),
        "spacing_pretrain_epochs": float(spacing_pretrain_epochs),
        "scaling_best_holdout": max(row["holdout_phoneme_best_guess"] for row in scaling_rows),
        "generalization_zero_pass_holdout": generalization["dictionary_pass_rows"][0]["holdout_phoneme_best_guess"],
        "generalization_last_pass_full": generalization["dictionary_pass_rows"][-1]["full_phoneme_best_guess"],
        "hard_c_count": float(generalization["rule_counts"]["hard_c"]),
        "soft_c_count": float(generalization["rule_counts"]["soft_c"]),
        "power_law_exponent": power_law["exponent"],
        "quantization_4bit_holdout": next(
            row["holdout_phoneme_best_guess"] for row in quantization_rows if int(row["bits"]) == 4
        ),
        "window_w7_final_train": float(w7_final_train),
        "window_w11_final_train": float(w11_final_train),
        "two_layer_pretrain_final_train": float(deep_train_final),
        "two_layer_pass0_full": float(100.0 * deep_pass0["full_phoneme_best_guess"]),
        "two_layer_pass11_full": float(100.0 * deep_pass11["full_phoneme_best_guess"]),
    }
    (results_dir / "repro_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Reproduction run complete.")
    print(f"Results: {results_dir}")
    print(f"Figures: {figs_dir}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
