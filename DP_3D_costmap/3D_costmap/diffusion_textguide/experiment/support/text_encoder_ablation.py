"""Experiment-only helpers for text encoder ablations."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch.utils.data import Dataset

from text_conditioning import (
    DEFAULT_FEATURE_DIMS,
    DEFAULT_MODEL_NAMES,
    FROZEN_FEATURE_ENCODERS,
    get_intent_to_id,
    is_frozen_feature_encoder,
    normalize_text_encoder_type,
)


class FrozenTextFeatureEncoder:
    """Frozen CLIP/BERT/T5 sentence encoder used only by ablation experiments."""

    def __init__(
        self,
        text_encoder_type: str,
        model_name: Optional[str] = None,
        device: Optional[torch.device] = None,
        batch_size: int = 64,
    ) -> None:
        self.text_encoder_type = normalize_text_encoder_type(text_encoder_type)
        if self.text_encoder_type not in FROZEN_FEATURE_ENCODERS:
            raise ValueError(f"{text_encoder_type!r} does not use frozen text features")

        self.model_name = model_name or DEFAULT_MODEL_NAMES[self.text_encoder_type]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        try:
            from transformers import (
                AutoModel,
                AutoTokenizer,
                CLIPTextModelWithProjection,
                CLIPTokenizer,
                T5EncoderModel,
            )
        except ImportError as exc:
            raise ImportError(
                "Frozen CLIP/BERT/T5 ablations require `transformers`. "
                "Install it in the training environment or use no_text/onehot."
            ) from exc

        if self.text_encoder_type in {"clip", "clip_proj"}:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
            self.model = CLIPTextModelWithProjection.from_pretrained(self.model_name)
        elif self.text_encoder_type == "t5_proj":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = T5EncoderModel.from_pretrained(self.model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)

        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.output_dim = int(DEFAULT_FEATURE_DIMS[self.text_encoder_type])
        hidden = getattr(getattr(self.model, "config", None), "hidden_size", None)
        d_model = getattr(getattr(self.model, "config", None), "d_model", None)
        projection_dim = getattr(getattr(self.model, "config", None), "projection_dim", None)
        if self.text_encoder_type in {"clip", "clip_proj"} and projection_dim is not None:
            self.output_dim = int(projection_dim)
        elif hidden is not None:
            self.output_dim = int(hidden)
        elif d_model is not None:
            self.output_dim = int(d_model)

    @torch.no_grad()
    def encode(self, sentences: List[str]) -> torch.Tensor:
        if not sentences:
            return torch.empty(0, self.output_dim)

        chunks = []
        for start in range(0, len(sentences), self.batch_size):
            batch = sentences[start:start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            if self.text_encoder_type in {"clip", "clip_proj"}:
                out = self.model(**encoded)
                feat = out.text_embeds
            elif self.text_encoder_type == "t5_proj":
                out = self.model(**encoded)
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                feat = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                out = self.model(**encoded)
                feat = getattr(out, "pooler_output", None)
                if feat is None:
                    feat = out.last_hidden_state[:, 0]

            chunks.append(feat.detach().cpu().float())

        return torch.cat(chunks, dim=0)


class TextEncoderAblationDataset(Dataset):
    """Adds experiment-specific condition fields to a metadata dataset."""

    def __init__(
        self,
        base_dataset: Dataset,
        text_encoder_type: str,
        feature_encoder: Optional[FrozenTextFeatureEncoder] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.text_encoder_type = normalize_text_encoder_type(text_encoder_type)
        self.intent_to_id = get_intent_to_id("train")
        self.vocab_size = getattr(base_dataset, "vocab_size", 0)
        self.text_feature_dim = 0
        self.text_features = None

        if is_frozen_feature_encoder(self.text_encoder_type):
            if feature_encoder is None:
                feature_encoder = FrozenTextFeatureEncoder(self.text_encoder_type)
            instructions = [base_dataset[i]["instruction"] for i in range(len(base_dataset))]
            unique = list(dict.fromkeys(instructions))
            encoded = feature_encoder.encode(unique)
            by_text = {s: encoded[i] for i, s in enumerate(unique)}
            self.text_features = torch.stack([by_text[s] for s in instructions])
            self.text_feature_dim = int(self.text_features.shape[-1])

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self.base_dataset[idx])
        if self.text_encoder_type == "onehot":
            intent_type = item.get("intent_type", "baseline")
            item["intent_id"] = torch.tensor(
                self.intent_to_id.get(intent_type, self.intent_to_id.get("baseline", 0)),
                dtype=torch.long,
            )
        if self.text_features is not None:
            item["text_feature"] = self.text_features[idx]
        return item
