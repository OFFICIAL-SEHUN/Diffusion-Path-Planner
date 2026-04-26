"""Utilities for loading instruction template splits from JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
DEFAULT_INSTRUCTION_DIR = ROOT / "data" / "instruction"


def load_instruction_templates(split: str = "train") -> Dict[str, List[str]]:
    """Load instruction templates for a split such as ``train`` or ``valid``."""
    path = DEFAULT_INSTRUCTION_DIR / split / f"inst_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Instruction template file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Instruction template file must contain an object: {path}")

    templates: Dict[str, List[str]] = {}
    for intent, sentences in data.items():
        if not isinstance(intent, str) or not isinstance(sentences, list):
            raise ValueError(f"Invalid instruction template entry for {intent!r} in {path}")
        if not all(isinstance(s, str) and s for s in sentences):
            raise ValueError(f"All instruction templates must be non-empty strings: {intent!r}")
        templates[intent] = sentences
    return templates
