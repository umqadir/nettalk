"""Data loading, encoding, and artifact generation for NETtalk experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constants import EXTRA_INPUT_SYMBOLS, GROUP_WIDTH, INPUT_INDEX, LETTER_SYMBOLS, OUTPUT_DIM


@dataclass(frozen=True)
class FeatureMapping:
    """Feature mapping for phoneme and stress encodings."""

    feature_indices: dict[str, int]
    stress_indices: dict[str, int]
    phoneme_features: dict[str, tuple[str, ...]]
    stress_encoding: dict[str, tuple[str, ...]]

    def vector_for(self, phoneme_symbol: str, stress_symbol: str) -> np.ndarray:
        vector = np.zeros(OUTPUT_DIM, dtype=np.float64)

        for feature_name in self.phoneme_features.get(phoneme_symbol, ()):
            if feature_name in self.feature_indices:
                vector[self.feature_indices[feature_name]] = 1.0

        for stress_feature in self.stress_encoding.get(stress_symbol, ()):
            if stress_feature in self.stress_indices:
                vector[self.stress_indices[stress_feature]] = 1.0
        return vector


@dataclass
class WordDataset:
    """Word-grouped data for NETtalk training with per-word updates."""

    words: list[str]
    phonemes: list[str]
    stresses: list[str]
    inputs: list[np.ndarray]
    targets: list[np.ndarray]
    window_size: int
    name: str

    @property
    def total_letters(self) -> int:
        return int(sum(array.shape[0] for array in self.inputs))

    @property
    def input_dim(self) -> int:
        if not self.inputs:
            return GROUP_WIDTH * self.window_size
        return int(self.inputs[0].shape[1])

    def flatten(self) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        if not self.inputs:
            return (
                np.zeros((0, self.input_dim), dtype=np.float64),
                np.zeros((0, OUTPUT_DIM), dtype=np.float64),
                [],
                [],
            )

        all_x = np.concatenate(self.inputs, axis=0)
        all_y = np.concatenate(self.targets, axis=0)
        phoneme_chars = [char for word in self.phonemes for char in word]
        stress_chars = [char for word in self.stresses for char in word]
        return all_x, all_y, phoneme_chars, stress_chars

    def repeat_words(self, repeats: int, name: str | None = None) -> "WordDataset":
        repeat_count = max(1, int(repeats))
        words = self.words * repeat_count
        phonemes = self.phonemes * repeat_count
        stresses = self.stresses * repeat_count
        inputs = [array.copy() for _ in range(repeat_count) for array in self.inputs]
        targets = [array.copy() for _ in range(repeat_count) for array in self.targets]
        return WordDataset(
            words=words,
            phonemes=phonemes,
            stresses=stresses,
            inputs=inputs,
            targets=targets,
            window_size=self.window_size,
            name=name or self.name,
        )

    def subset(self, indices: list[int], name: str | None = None) -> "WordDataset":
        return WordDataset(
            words=[self.words[idx] for idx in indices],
            phonemes=[self.phonemes[idx] for idx in indices],
            stresses=[self.stresses[idx] for idx in indices],
            inputs=[self.inputs[idx].copy() for idx in indices],
            targets=[self.targets[idx].copy() for idx in indices],
            window_size=self.window_size,
            name=name or self.name,
        )


def _repo_root(explicit_root: str | Path | None = None) -> Path:
    if explicit_root is not None:
        return Path(explicit_root).resolve()
    return Path(__file__).resolve().parent.parent


def parse_nettalk_data_file(data_path: str | Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with Path(data_path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            word, phonemes, stress, flag_text = parts
            if not word.isalpha():
                continue
            if not (len(word) == len(phonemes) == len(stress)):
                continue
            entries.append(
                {
                    "word": word.lower(),
                    "phonemes": phonemes,
                    "stress": stress,
                    "flag": int(flag_text),
                }
            )
    return entries


def load_nettalk_entries(repo_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = _repo_root(repo_root)
    parsed_json_path = root / "data" / "processed" / "nettalk_parsed.json"
    raw_data_path = root / "data" / "raw" / "nettalk.data"

    # Treat the raw corpus as authoritative for all reproduction runs. The parsed
    # JSON is only a fallback for local exploration when the raw file is absent.
    if raw_data_path.exists():
        parsed_entries = parse_nettalk_data_file(raw_data_path)
    elif parsed_json_path.exists():
        loaded = json.loads(parsed_json_path.read_text(encoding="utf-8"))
        parsed_entries = [
            {
                "word": item["word"].lower(),
                "phonemes": item["phonemes"],
                "stress": item["stress"],
                "flag": int(item["flag"]),
            }
            for item in loaded
            if len(item["word"]) == len(item["phonemes"]) == len(item["stress"])
        ]
    else:
        raise FileNotFoundError(f"Missing NETtalk corpus at {raw_data_path}")

    # The benchmark documentation for NETtalk notes that some words appear more than
    # once with different pronunciations and that the original experiments used only
    # the first pronunciation for each word. We mirror that by keeping the first
    # occurrence in the source ordering.
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for entry in parsed_entries:
        word = entry["word"]
        if word in seen:
            continue
        seen.add(word)
        deduped.append(entry)
    return deduped


def load_feature_mapping(repo_root: str | Path | None = None) -> FeatureMapping:
    root = _repo_root(repo_root)
    mapping_path = root / "data" / "processed" / "phoneme_features.json"
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))

    phoneme_features = {
        symbol: tuple(item.get("features", []))
        for symbol, item in payload["phonemes"].items()
    }
    stress_encoding = {
        symbol: tuple(item.get("features", []))
        for symbol, item in payload["stress_encoding"].items()
    }

    return FeatureMapping(
        feature_indices={key: int(value) for key, value in payload["_feature_indices"].items()},
        stress_indices={key: int(value) for key, value in payload["_stress_indices"].items()},
        phoneme_features=phoneme_features,
        stress_encoding=stress_encoding,
    )


def load_common_1000_words(repo_root: str | Path | None = None) -> list[str]:
    root = _repo_root(repo_root)
    word_list_path = root / "data" / "raw" / "nettalk_1000_words.txt"
    return [line.strip().lower() for line in word_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _encode_input_symbol(char: str) -> int:
    lowered = char.lower()
    if lowered in INPUT_INDEX:
        return INPUT_INDEX[lowered]
    if lowered in LETTER_SYMBOLS:
        return INPUT_INDEX[lowered]
    if lowered == ".":
        return INPUT_INDEX["."]
    return INPUT_INDEX["'"]


def encode_window(word: str, center_index: int, window_size: int) -> np.ndarray:
    radius = window_size // 2
    vector = np.zeros((window_size, GROUP_WIDTH), dtype=np.float64)

    for offset, absolute_index in enumerate(range(center_index - radius, center_index + radius + 1)):
        if absolute_index < 0 or absolute_index >= len(word):
            symbol = EXTRA_INPUT_SYMBOLS[0]
        else:
            symbol = word[absolute_index]
        vector[offset, _encode_input_symbol(symbol)] = 1.0

    return vector.reshape(window_size * GROUP_WIDTH)


def build_dataset(
    entries: list[dict[str, Any]],
    mapping: FeatureMapping,
    *,
    name: str,
    window_size: int,
) -> WordDataset:
    words: list[str] = []
    phonemes: list[str] = []
    stresses: list[str] = []
    all_inputs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for entry in entries:
        word = entry["word"].lower()
        phoneme_word = entry["phonemes"]
        stress_word = entry["stress"]
        if not (len(word) == len(phoneme_word) == len(stress_word)):
            continue

        input_rows = np.zeros((len(word), window_size * GROUP_WIDTH), dtype=np.float64)
        output_rows = np.zeros((len(word), OUTPUT_DIM), dtype=np.float64)
        for index in range(len(word)):
            input_rows[index] = encode_window(word, index, window_size=window_size)
            output_rows[index] = mapping.vector_for(phoneme_word[index], stress_word[index])

        words.append(word)
        phonemes.append(phoneme_word)
        stresses.append(stress_word)
        all_inputs.append(input_rows)
        all_targets.append(output_rows)

    return WordDataset(
        words=words,
        phonemes=phonemes,
        stresses=stresses,
        inputs=all_inputs,
        targets=all_targets,
        window_size=window_size,
        name=name,
    )


def _entry_dict(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {entry["word"]: entry for entry in entries}


def save_preprocessed_dataset(dataset: WordDataset, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lengths = np.array([array.shape[0] for array in dataset.inputs], dtype=np.int32)
    flattened_inputs, flattened_targets, _, _ = dataset.flatten()
    np.savez_compressed(
        output_path,
        name=dataset.name,
        window_size=np.int32(dataset.window_size),
        lengths=lengths,
        words=np.array(dataset.words, dtype=object),
        phonemes=np.array(dataset.phonemes, dtype=object),
        stresses=np.array(dataset.stresses, dtype=object),
        inputs=flattened_inputs,
        targets=flattened_targets,
    )


def load_preprocessed_dataset(path: str | Path) -> WordDataset:
    payload = np.load(path, allow_pickle=True)
    lengths = payload["lengths"].astype(np.int32).tolist()
    words = payload["words"].astype(object).tolist()
    phonemes = payload["phonemes"].astype(object).tolist()
    stresses = payload["stresses"].astype(object).tolist()
    inputs_flat = payload["inputs"]
    targets_flat = payload["targets"]
    window_size = int(payload["window_size"])
    name = str(payload["name"])

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    start = 0
    for length in lengths:
        end = start + length
        inputs.append(inputs_flat[start:end].copy())
        targets.append(targets_flat[start:end].copy())
        start = end

    return WordDataset(
        words=words,
        phonemes=phonemes,
        stresses=stresses,
        inputs=inputs,
        targets=targets,
        window_size=window_size,
        name=name,
    )


def build_repro_artifacts(
    repo_root: str | Path | None = None,
    *,
    window_sizes: tuple[int, ...] = (3, 5, 7, 9, 11),
) -> dict[str, Any]:
    root = _repo_root(repo_root)
    mapping = load_feature_mapping(root)
    all_entries = load_nettalk_entries(root)
    common_words = load_common_1000_words(root)

    entry_by_word = _entry_dict(all_entries)
    train_1000_entries = [entry_by_word[word] for word in common_words if word in entry_by_word]
    common_word_set = {entry["word"] for entry in train_1000_entries}
    holdout_entries = [entry for entry in all_entries if entry["word"] not in common_word_set]

    out_root = root / "data" / "processed" / "repro"
    out_root.mkdir(parents=True, exist_ok=True)

    datasets_created: dict[str, Any] = {}
    for window_size in window_sizes:
        train_dataset = build_dataset(
            train_1000_entries,
            mapping,
            name=f"train_1000_w{window_size}",
            window_size=window_size,
        )
        holdout_dataset = build_dataset(
            holdout_entries,
            mapping,
            name=f"holdout_dict_w{window_size}",
            window_size=window_size,
        )
        full_dataset = build_dataset(
            all_entries,
            mapping,
            name=f"full_dict_w{window_size}",
            window_size=window_size,
        )

        datasets = {
            train_dataset.name: train_dataset,
            holdout_dataset.name: holdout_dataset,
            full_dataset.name: full_dataset,
        }
        for dataset_name, dataset in datasets.items():
            save_preprocessed_dataset(dataset, out_root / f"{dataset_name}.npz")

        datasets_created[str(window_size)] = {
            "train_words": len(train_dataset.words),
            "holdout_words": len(holdout_dataset.words),
            "full_words": len(full_dataset.words),
            "train_letters": train_dataset.total_letters,
            "holdout_letters": holdout_dataset.total_letters,
            "full_letters": full_dataset.total_letters,
        }

    manifest = {
        "decision": {
            "primary_paper_spec": "1987_journal",
            "output_encoding": "21_articulatory_plus_5_stress_boundary",
            "informal_speech_status": "deferred_secondary",
            "duplicate_pronunciations": "keep_first_per_word",
        },
        "counts": {
            "all_words": len(all_entries),
            "train_1000_words": len(train_1000_entries),
            "holdout_words": len(holdout_entries),
        },
        "window_sizes": datasets_created,
        "input_symbols": LETTER_SYMBOLS + EXTRA_INPUT_SYMBOLS,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
