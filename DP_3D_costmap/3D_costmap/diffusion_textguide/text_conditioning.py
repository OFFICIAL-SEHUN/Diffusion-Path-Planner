"""Shared text-conditioning identifiers used by model and experiments."""

from __future__ import annotations

from typing import Dict, Optional

from instruction_utils import load_instruction_templates


TEXT_ENCODER_ALIASES = {
    "none": "no_text",
    "no_text": "no_text",
    "notext": "no_text",
    "onehot": "onehot",
    "onehot_intent": "onehot",
    "one-hot": "onehot",
    "clip": "clip",
    "frozen_clip": "clip",
    "clip_proj": "clip_proj",
    "frozen_clip_proj": "clip_proj",
    "bert": "bert_proj",
    "bert_proj": "bert_proj",
    "frozen_bert_proj": "bert_proj",
    "t5": "t5_proj",
    "t5_proj": "t5_proj",
    "frozen_t5_proj": "t5_proj",
    "learnable": "learnable",
}

FROZEN_FEATURE_ENCODERS = {"clip", "clip_proj", "bert_proj", "t5_proj"}
PROJECTED_FEATURE_ENCODERS = {"clip_proj", "bert_proj", "t5_proj"}

DEFAULT_MODEL_NAMES = {
    "clip": "openai/clip-vit-base-patch32",
    "clip_proj": "openai/clip-vit-base-patch32",
    "bert_proj": "bert-base-uncased",
    "t5_proj": "t5-base",
}

DEFAULT_FEATURE_DIMS = {
    "clip": 512,
    "clip_proj": 512,
    "bert_proj": 768,
    "t5_proj": 768,
}


def normalize_text_encoder_type(text_encoder_type: Optional[str]) -> str:
    key = (text_encoder_type or "learnable").strip().lower()
    if key not in TEXT_ENCODER_ALIASES:
        valid = ", ".join(sorted(TEXT_ENCODER_ALIASES))
        raise ValueError(f"Unknown text_encoder_type={text_encoder_type!r}. Valid aliases: {valid}")
    return TEXT_ENCODER_ALIASES[key]


def is_frozen_feature_encoder(text_encoder_type: str) -> bool:
    return normalize_text_encoder_type(text_encoder_type) in FROZEN_FEATURE_ENCODERS


def uses_trainable_projection(text_encoder_type: str) -> bool:
    return normalize_text_encoder_type(text_encoder_type) in PROJECTED_FEATURE_ENCODERS


def get_intent_to_id(split: str = "train") -> Dict[str, int]:
    return {intent: i for i, intent in enumerate(load_instruction_templates(split).keys())}
