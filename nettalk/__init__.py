"""NETtalk reproduction package."""

from .constants import INPUT_DIM, OUTPUT_DIM, WINDOW_SIZE_DEFAULT
from .data import (
    FeatureMapping,
    WordDataset,
    build_repro_artifacts,
    load_feature_mapping,
    load_preprocessed_dataset,
)
from .eval import evaluate_dataset
from .experiments import (
    run_damage_and_relearning_experiment,
    run_generalization_experiment,
    run_hidden_clustering_experiment,
    run_hidden_scaling_experiment,
    run_quantization_experiment,
    run_spacing_experiment,
)
from .model import NettalkMLP
from .train import TrainingConfig, train_model

__all__ = [
    "FeatureMapping",
    "INPUT_DIM",
    "NettalkMLP",
    "OUTPUT_DIM",
    "TrainingConfig",
    "WINDOW_SIZE_DEFAULT",
    "WordDataset",
    "build_repro_artifacts",
    "evaluate_dataset",
    "load_feature_mapping",
    "load_preprocessed_dataset",
    "run_damage_and_relearning_experiment",
    "run_generalization_experiment",
    "run_hidden_clustering_experiment",
    "run_hidden_scaling_experiment",
    "run_quantization_experiment",
    "run_spacing_experiment",
    "train_model",
]
