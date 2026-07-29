"""Shared constants for NETtalk reproduction."""

from __future__ import annotations

import string

LETTER_SYMBOLS = list(string.ascii_lowercase)
EXTRA_INPUT_SYMBOLS = ["_", ".", "'"]
INPUT_SYMBOLS = LETTER_SYMBOLS + EXTRA_INPUT_SYMBOLS
INPUT_INDEX = {symbol: idx for idx, symbol in enumerate(INPUT_SYMBOLS)}

WINDOW_SIZE_DEFAULT = 7
WINDOW_SIZE_MODERN = 11

GROUP_WIDTH = len(INPUT_SYMBOLS)
INPUT_DIM = GROUP_WIDTH * WINDOW_SIZE_DEFAULT
OUTPUT_DIM = 26

PHONEME_FEATURE_DIM = 21
STRESS_FEATURE_DIM = 5
