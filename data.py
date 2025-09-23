"""Straightforward dataset helpers used across the training scripts.

The goal of this module is to keep token handling approachable.  Instead of
having custom collate functions that juggle padding and masks, each dataset
returns ready-to-batch tensors.  The default PyTorch :class:`~torch.utils.data.DataLoader`
can therefore be used without any bells and whistles which mirrors the style of
NanoGPT's educational implementation.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

import tiktoken


@dataclass
class TokenizerInfo:
    """Tokenizer helper exposing the encoder and its special token ids."""

    encoder: object
    bos: int
    eos: int
    pad: int

    def encode(self, text: str) -> list[int]:
        return self.encoder.encode(text)


def build_tokenizer() -> TokenizerInfo:
    """Create a GPT-2 compatible tokenizer or fall back to a byte-level one."""

    try:
        enc = tiktoken.get_encoding("gpt2")
        bos = enc.eot_token
        eos = enc.eot_token
        pad = enc.eot_token
        return TokenizerInfo(enc, bos, eos, pad)
    except Exception as exc:  # pragma: no cover - offline fallback
        warnings.warn(
            "Falling back to a minimal byte-level tokenizer because the GPT-2 "
            f"encoding could not be loaded ({exc}).",
            RuntimeWarning,
        )
        byte_tokens = {bytes([i]): i for i in range(256)}
        bos_id = len(byte_tokens)
        eos_id = bos_id + 1
        pad_id = eos_id + 1
        special_tokens = {"<|bos|>": bos_id, "<|eos|>": eos_id, "<|pad|>": pad_id}
        enc = tiktoken.Encoding(
            "byte-level-minimal",
            pat_str=r"(?s).",
            mergeable_ranks=byte_tokens,
            special_tokens=special_tokens,
        )
        bos = special_tokens["<|bos|>"]
        eos = special_tokens["<|eos|>"]
        pad = special_tokens["<|pad|>"]
        return TokenizerInfo(enc, bos, eos, pad)


def load_jsonl(path: str | Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _split_into_sequences(tokens: list[int], block_size: int) -> Iterable[list[int]]:
    """Yield overlapping chunks that are at most ``block_size + 1`` long."""

    step = block_size + 1
    for start in range(0, max(0, len(tokens) - 1), block_size):
        yield tokens[start : start + step]


def _pack(sequence: list[int], *, pad_value: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad ``sequence`` to ``length`` and return both the tensor and its mask."""

    tensor = torch.full((length,), pad_value, dtype=torch.long)
    mask = torch.zeros(length, dtype=torch.long)
    valid = min(len(sequence), length)
    if valid:
        tensor[:valid] = torch.tensor(sequence[:length], dtype=torch.long)
        mask[:valid] = 1
    return tensor, mask


class SupervisedDataset(Dataset):
    """Language modelling dataset mirroring NanoGPT's simple formulation."""

    def __init__(self, path: str | Path, block_size: int) -> None:
        self.tokenizer = build_tokenizer()
        self.block_size = block_size
        self._rows: list[dict[str, torch.Tensor]] = []

        for row in load_jsonl(path):
            prompt = row.get("prompt", "")
            answer = row.get("chosen", row.get("response", ""))

            prompt_tokens = [self.tokenizer.bos] + self.tokenizer.encode(prompt)
            tokens = prompt_tokens + self.tokenizer.encode(answer) + [self.tokenizer.eos]

            for chunk in _split_into_sequences(tokens, block_size):
                if len(chunk) < 2:
                    continue

                inputs = chunk[:-1]
                targets = chunk[1:]

                input_tensor, attention_mask = _pack(
                    inputs, pad_value=self.tokenizer.pad, length=block_size
                )
                target_tensor, _ = _pack(
                    targets, pad_value=self.tokenizer.pad, length=block_size
                )
                target_tensor[attention_mask == 0] = -1

                label_mask = attention_mask.clone()

                self._rows.append(
                    {
                        "input_ids": input_tensor,
                        "target_ids": target_tensor,
                        "attention_mask": attention_mask,
                        "label_mask": label_mask,
                    }
                )

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._rows[index]


class PreferenceDataset(Dataset):
    """Preference data where every row already carries its padding mask."""

    def __init__(self, path: str | Path, block_size: int) -> None:
        self.tokenizer = build_tokenizer()
        self.block_size = block_size
        self._rows: list[dict[str, torch.Tensor]] = []

        for row in load_jsonl(path):
            prompt = row.get("prompt", "")
            chosen = row.get("chosen", "")
            rejected = row.get("rejected", "")

            prompt_tokens = [self.tokenizer.bos] + self.tokenizer.encode(prompt)
            chosen_tokens = prompt_tokens + self.tokenizer.encode(chosen) + [self.tokenizer.eos]
            rejected_tokens = prompt_tokens + self.tokenizer.encode(rejected) + [self.tokenizer.eos]

            prompt_tensor, prompt_mask = _pack(
                prompt_tokens, pad_value=self.tokenizer.pad, length=block_size
            )
            chosen_tensor, chosen_mask = _pack(
                chosen_tokens, pad_value=self.tokenizer.pad, length=block_size
            )
            rejected_tensor, rejected_mask = _pack(
                rejected_tokens, pad_value=self.tokenizer.pad, length=block_size
            )

            self._rows.append(
                {
                    "prompt": prompt_tensor,
                    "prompt_mask": prompt_mask,
                    "chosen": chosen_tensor,
                    "chosen_mask": chosen_mask,
                    "rejected": rejected_tensor,
                    "rejected_mask": rejected_mask,
                }
            )

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._rows[index]
