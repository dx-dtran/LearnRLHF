import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

try:  # pragma: no cover - optional dependency
    import tiktoken
except Exception:  # pragma: no cover - fallback when tiktoken is unavailable
    tiktoken = None


class _FallbackEncoding:
    """Extremely small tokenizer used when ``tiktoken`` is missing.

    Tokens are whitespace separated so that the repository can run without any
    external downloads.  It only keeps track of the tokens that appear in the
    loaded dataset which mirrors the behaviour of classic toy language models.
    """

    def __init__(self) -> None:
        self.stoi: dict[str, int] = {"<pad>": 0, "<eos>": 1}
        self.itos: list[str] = ["<pad>", "<eos>"]

    @property
    def bos_token(self) -> int:
        return 1

    @property
    def eos_token(self) -> int:
        return 1

    def encode(self, text: str) -> list[int]:
        tokens = []
        for piece in text.split():
            if piece not in self.stoi:
                self.stoi[piece] = len(self.itos)
                self.itos.append(piece)
            tokens.append(self.stoi[piece])
        return tokens


@dataclass
class TokenizerBundle:
    encoder: object
    bos: int
    eos: int
    pad: int

    def encode(self, text: str) -> list[int]:
        return self.encoder.encode(text)


def build_tokenizer() -> TokenizerBundle:
    if tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("gpt2")
        except Exception:
            enc = None
        else:
            bos = enc._special_tokens.get("<|bos|>", enc.eot_token)
            eos = enc.eot_token
            pad = eos
            return TokenizerBundle(enc, bos, eos, pad)
    fallback = _FallbackEncoding()
    return TokenizerBundle(fallback, fallback.bos_token, fallback.eos_token, 0)


def load_jsonl(path: str | Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _split_into_sequences(tokens: list[int], block_size: int) -> Iterable[list[int]]:
    step = block_size + 1
    for start in range(0, max(0, len(tokens) - 1), block_size):
        yield tokens[start : start + step]


class SupervisedDataset(Dataset):
    """Language modelling dataset with explicit ``x``/``y`` pairs.

    Each JSONL row must contain ``prompt`` and ``chosen`` (or ``response``).
    The input sequence ``x`` is every token except the last one while ``y`` is
    the same sequence shifted by one position, mirroring the formulation used in
    Andrej Karpathy's micro GPT implementations.
    """

    def __init__(self, path: str | Path, block_size: int) -> None:
        self.bundle = build_tokenizer()
        self.block_size = block_size
        self.data: list[tuple[torch.Tensor, torch.Tensor]] = []

        for row in load_jsonl(path):
            prompt = row.get("prompt", "")
            answer = row.get("chosen", row.get("response", ""))
            text = prompt + answer
            tokens = [self.bundle.bos] + self.bundle.encode(text) + [self.bundle.eos]
            for chunk in _split_into_sequences(tokens, block_size):
                if len(chunk) < 2:
                    continue
                x_tokens = torch.tensor(chunk[:-1], dtype=torch.long)
                y_tokens = torch.tensor(chunk[1:], dtype=torch.long)
                self.data.append((x_tokens, y_tokens))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index]


class PreferenceDataset(Dataset):
    """Dataset used for training reward models and PPO."""

    def __init__(self, path: str | Path, block_size: int) -> None:
        self.bundle = build_tokenizer()
        self.block_size = block_size
        self.rows: list[dict[str, torch.Tensor]] = []

        for row in load_jsonl(path):
            prompt = row.get("prompt", "")
            chosen = row.get("chosen", "")
            rejected = row.get("rejected", "")

            prompt_tokens = [self.bundle.bos] + self.bundle.encode(prompt)
            chosen_tokens = prompt_tokens + self.bundle.encode(chosen) + [self.bundle.eos]
            rejected_tokens = (
                prompt_tokens + self.bundle.encode(rejected) + [self.bundle.eos]
            )

            self.rows.append(
                {
                    "prompt": torch.tensor(prompt_tokens[:block_size], dtype=torch.long),
                    "chosen": torch.tensor(chosen_tokens[:block_size], dtype=torch.long),
                    "rejected": torch.tensor(rejected_tokens[:block_size], dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.rows[index]


def _pad_sequences(seqs: list[torch.Tensor], pad_value: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not seqs:
        return torch.zeros(0, 0, dtype=torch.long), torch.zeros(0, 0, dtype=torch.long)
    max_len = max(seq.size(0) for seq in seqs)
    batch = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, seq in enumerate(seqs):
        length = seq.size(0)
        batch[i, :length] = seq
        mask[i, :length] = 1
    return batch, mask


def collate_supervised(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_token: int) -> dict[str, torch.Tensor]:
    inputs, targets = zip(*batch)
    x, mask = _pad_sequences(list(inputs), pad_token)
    y, _ = _pad_sequences(list(targets), -1)
    y = y.long()
    y[mask == 0] = -1
    return {"input_ids": x, "target_ids": y, "attention_mask": mask}


def collate_preferences(batch: list[dict[str, torch.Tensor]], pad_token: int) -> dict[str, torch.Tensor]:
    prompts = [item["prompt"] for item in batch]
    chosen = [item["chosen"] for item in batch]
    rejected = [item["rejected"] for item in batch]

    prompt_pad, prompt_mask = _pad_sequences(prompts, pad_token)
    chosen_pad, chosen_mask = _pad_sequences(chosen, pad_token)
    rejected_pad, rejected_mask = _pad_sequences(rejected, pad_token)

    return {
        "prompt": prompt_pad,
        "prompt_mask": prompt_mask,
        "chosen": chosen_pad,
        "chosen_mask": chosen_mask,
        "rejected": rejected_pad,
        "rejected_mask": rejected_mask,
    }
