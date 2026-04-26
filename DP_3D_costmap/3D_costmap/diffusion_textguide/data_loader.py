"""
TextGuide Dataset & Vocabulary

.pt 파일 (intent 포맷)에서 (costmap, path, text_tokens, start, goal)을 로드.
JSON instruction templates로부터 word-level vocab을 자동 빌드.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from instruction_utils import load_instruction_templates


INSTRUCTION_TEMPLATES = load_instruction_templates("train")


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocab(extra_sentences: Optional[List[str]] = None) -> Dict[str, int]:
    """INSTRUCTION_TEMPLATES + 추가 문장에서 word-level vocab 빌드."""
    words = set()
    for templates in INSTRUCTION_TEMPLATES.values():
        for s in templates:
            for w in s.lower().split():
                words.add(w)
    if extra_sentences:
        for s in extra_sentences:
            for w in s.lower().split():
                words.add(w)
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for i, w in enumerate(sorted(words), start=2):
        vocab[w] = i
    return vocab


def text_to_tokens(text: str, vocab: Dict[str, int],
                   max_seq_len: int = 16) -> torch.Tensor:
    """텍스트를 token ID 시퀀스로 변환 (pad/truncate)."""
    ids = []
    for w in text.lower().split():
        ids.append(vocab.get(w, vocab[UNK_TOKEN]))
    if len(ids) > max_seq_len:
        ids = ids[:max_seq_len]
    else:
        ids += [vocab[PAD_TOKEN]] * (max_seq_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


class TextGuideDataset(Dataset):
    """Intent 기반 .pt 파일을 로드하여 flat (costmap, path, tokens) 샘플을 제공."""

    def __init__(self, data_dir: str, max_seq_len: int = 16,
                 vocab: Optional[Dict[str, int]] = None,
                 return_metadata: bool = False):
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.vocab = vocab if vocab is not None else build_vocab()
        self.vocab_size = len(self.vocab)
        self.return_metadata = return_metadata

        pt_files = sorted(self.data_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files in {data_dir}")

        # 1차 스캔: 데이터 파일의 instruction에서 추가 vocab 수집
        all_instructions = []
        raw_data = []
        for f in pt_files:
            data = torch.load(f, map_location="cpu", weights_only=False)
            raw_data.append(data)
            for instr in data.get("instructions", []):
                all_instructions.append(instr)

        if self.vocab is None or len(self.vocab) <= 2:
            self.vocab = build_vocab(extra_sentences=all_instructions)
        else:
            extra = build_vocab(extra_sentences=all_instructions)
            for w, idx in extra.items():
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab)
        self.vocab_size = len(self.vocab)

        # 2차 스캔: 실제 데이터 로드
        self.costmaps = []
        self.paths = []
        self.tokens = []
        self.instructions = []
        self.intent_types = []

        for data in raw_data:
            costmap = data["costmap"]                # [2, H, W]
            paths = data["paths"]                    # [N, horizon, 2]
            instructions = data.get("instructions", [])
            intent_types = data.get("intent_types", [])
            n_paths = paths.shape[0]

            for i in range(n_paths):
                self.costmaps.append(costmap)
                self.paths.append(paths[i])
                instr = instructions[i] if i < len(instructions) else ""
                intent_type = intent_types[i] if i < len(intent_types) else "baseline"
                self.tokens.append(text_to_tokens(instr, self.vocab, max_seq_len))
                self.instructions.append(instr)
                self.intent_types.append(intent_type)

        self.costmaps = torch.stack(self.costmaps)   # [N_total, 2, H, W]
        self.paths = torch.stack(self.paths)          # [N_total, horizon, 2]
        self.tokens = torch.stack(self.tokens)        # [N_total, max_seq_len]

        print(f"Loaded {len(self)} samples from {len(pt_files)} files "
              f"(vocab_size={self.vocab_size})")

    def __len__(self) -> int:
        return self.paths.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.return_metadata:
            return {
                "costmap": self.costmaps[idx],
                "path": self.paths[idx],
                "tokens": self.tokens[idx],
                "instruction": self.instructions[idx],
                "intent_type": self.intent_types[idx],
            }
        return self.costmaps[idx], self.paths[idx], self.tokens[idx]
