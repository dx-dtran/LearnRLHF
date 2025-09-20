import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset
import tiktoken


class SimpleEncoding:
    def __init__(self):
        self.vocab = {"<pad>": 0, "<eos>": 1}
        self.eot_token = 1
        self._special_tokens = {"<|bos|>": 1}

    def encode(self, text: str) -> list[int]:
        tokens = []
        for token in text.split():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
            tokens.append(self.vocab[token])
        return tokens


def load_jsonl(path: str | Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


@dataclass
class TokenizerBundle:
    tokenizer: tiktoken.Encoding
    bos: int
    eos: int
    pad: int


def build_tokenizer() -> TokenizerBundle:
    try:
        enc = tiktoken.get_encoding("gpt2")
        bos = enc._special_tokens.get("<|bos|>", enc.eot_token)
        eos = enc.eot_token
        pad = eos
        return TokenizerBundle(enc, bos, eos, pad)
    except Exception:
        enc = SimpleEncoding()
        return TokenizerBundle(enc, enc._special_tokens["<|bos|>"], enc.eot_token, 0)


def encode_with_specials(bundle: TokenizerBundle, text: str) -> list[int]:
    tokens = bundle.tokenizer.encode(text)
    return tokens


class SupervisedDataset(Dataset):
    def __init__(self, path: str | Path, block_size: int):
        self.bundle = build_tokenizer()
        self.block_size = block_size
        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []

        for row in load_jsonl(path):
            prompt = row.get("prompt", "")
            response = row.get("chosen", row.get("response", ""))
            text = prompt + response
            ids = [self.bundle.bos] + encode_with_specials(self.bundle, text) + [
                self.bundle.eos
            ]
            if len(ids) < 2:
                continue
            start = 0
            while start + 1 < len(ids):
                chunk = ids[start : start + self.block_size + 1]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                self.examples.append((x, y))
                start += self.block_size

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[idx]


class PreferenceDataset(Dataset):
    def __init__(self, path: str | Path, block_size: int):
        self.bundle = build_tokenizer()
        self.block_size = block_size
        self.samples: list[dict[str, torch.Tensor]] = []

        for row in load_jsonl(path):
            prompt = row.get("prompt", "")
            chosen = row.get("chosen", "")
            rejected = row.get("rejected", "")

            prompt_ids = [self.bundle.bos] + encode_with_specials(self.bundle, prompt)
            chosen_tail = encode_with_specials(self.bundle, chosen) + [self.bundle.eos]
            rejected_tail = (
                encode_with_specials(self.bundle, rejected) + [self.bundle.eos]
            )

            prompt_ids = prompt_ids[: self.block_size]
            chosen_ids = (prompt_ids + chosen_tail)[: self.block_size]
            rejected_ids = (prompt_ids + rejected_tail)[: self.block_size]

            self.samples.append(
                {
                    "prompt": torch.tensor(prompt_ids, dtype=torch.long),
                    "chosen": torch.tensor(chosen_ids, dtype=torch.long),
                    "rejected": torch.tensor(rejected_ids, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


def pad_batch(seqs: list[torch.Tensor], pad_token: int, pad_value: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(seq.size(0) for seq in seqs)
    out = torch.full((len(seqs), max_len), pad_token, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, seq in enumerate(seqs):
        length = seq.size(0)
        out[i, :length] = seq
        mask[i, :length] = 1
    return out, mask


def collate_supervised(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_token: int) -> dict[str, torch.Tensor]:
    inputs, targets = zip(*batch)
    x, mask = pad_batch(list(inputs), pad_token)
    y, _ = pad_batch(list(targets), -1)
    y[y == -1] = -1
    for i, seq in enumerate(targets):
        y[i, : seq.size(0)] = seq
        if seq.size(0) < y.size(1):
            y[i, seq.size(0) :] = -1
    return {"input_ids": x, "target_ids": y, "attention_mask": mask}


def collate_preferences(batch: list[dict[str, torch.Tensor]], pad_token: int) -> dict[str, torch.Tensor]:
    prompts = [item["prompt"] for item in batch]
    chosen = [item["chosen"] for item in batch]
    rejected = [item["rejected"] for item in batch]

    prompt_pad, prompt_mask = pad_batch(prompts, pad_token)
    chosen_pad, chosen_mask = pad_batch(chosen, pad_token)
    rejected_pad, rejected_mask = pad_batch(rejected, pad_token)

    return {
        "prompt": prompt_pad,
        "prompt_mask": prompt_mask,
        "chosen": chosen_pad,
        "chosen_mask": chosen_mask,
        "rejected": rejected_pad,
        "rejected_mask": rejected_mask,
    }
