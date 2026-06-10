"""
data_hh.py — Anthropic HH-RLHF download + the three derived datasets.

This file is fully provided. It is plumbing, not the lesson — but read it once: the
collate functions decide what your losses see, and two of the course's classic bugs
(label shifting, left vs right padding) live here.

Three datasets:

    1) SFTDataset / sft_collate          -> input_ids, labels, loss_mask, attention_mask
       from the *chosen* dialogue, loss only on assistant tokens. RIGHT-padded.
    2) PreferenceDataset / rm_collate    -> (chosen, rejected) tokenizations + the index
       of each sequence's last real token, where the reward head pools. RIGHT-padded.
    3) PromptDataset / prompt_collate    -> prompt-only token ids for PPO rollouts.
       LEFT-padded, so every row's last column is the position generation grows from.

Raw HH rows are {"chosen": str, "rejected": str}; both strings look like

    \\n\\nHuman: ...\\n\\nAssistant: ...\\n\\nHuman: ...\\n\\nAssistant: ...

and differ only in the final Assistant turn.
"""

from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from tokenizer import EOT_ID, build_sft_example, encode, format_chat, format_prompt


# -------------------------------------------------------------------------------------
# Download + parsing
# -------------------------------------------------------------------------------------


def parse_hh_dialogue(text: str) -> List[dict]:
    """
    Parse a raw HH dialogue string into [{"role": ..., "content": ...}, ...].

    Splits on the literal markers "\\n\\nHuman:" / "\\n\\nAssistant:". Tolerates
    leading whitespace and (rare) dialogues that open with Assistant.
    """
    turns: List[dict] = []
    text = "\n\n" + text.strip()
    # Walk the string, finding each marker and the start of the next one.
    markers = [("\n\nHuman:", "user"), ("\n\nAssistant:", "assistant")]
    i = 0
    current_role = None
    current_start = None
    while i < len(text):
        hit = None
        for marker, role in markers:
            if text.startswith(marker, i):
                hit = (marker, role)
                break
        if hit is None:
            i += 1
            continue
        if current_role is not None:
            content = text[current_start:i].strip()
            if content:
                turns.append({"role": current_role, "content": content})
        current_role = hit[1]
        current_start = i + len(hit[0])
        i = current_start
    if current_role is not None:
        content = text[current_start:].strip()
        if content:
            turns.append({"role": current_role, "content": content})
    return turns


def download_hh(split: str = "train", cache_dir: str = "hh_cache") -> List[dict]:
    """Download (and cache) Anthropic/hh-rlhf as a list of {"chosen","rejected"} dicts."""
    from datasets import load_dataset  # local import: tests never need `datasets`

    ds = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
    return [{"chosen": r["chosen"], "rejected": r["rejected"]} for r in ds]


def split_prompt_and_last_response(dialogue: List[dict]) -> Tuple[List[dict], str]:
    """Everything except the final assistant message, plus that message's content."""
    assert dialogue and dialogue[-1]["role"] == "assistant", (
        "HH dialogues must end with an assistant turn"
    )
    return dialogue[:-1], dialogue[-1]["content"]


# -------------------------------------------------------------------------------------
# 1) SFT dataset — chosen dialogues, assistant-only loss mask
# -------------------------------------------------------------------------------------


class SFTDataset(Dataset):
    def __init__(self, hh_rows: List[dict], block_size: int = 1024):
        self.block_size = block_size
        self.examples: List[Tuple[List[int], List[int]]] = []
        for row in hh_rows:
            turns = parse_hh_dialogue(row["chosen"])
            if not turns:
                continue
            ids, mask = build_sft_example(turns)
            ids, mask = ids[:block_size], mask[:block_size]
            if sum(mask) == 0:
                continue  # truncation removed all assistant supervision
            self.examples.append((ids, mask))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        input_ids, loss_mask = self.examples[i]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float32),
        }


def sft_collate(batch: List[dict]) -> dict:
    """
    Right-pad to the longest sequence in the batch and build shifted labels.

    At position t the model predicts token t+1, so labels[t] = input_ids[t+1] and the
    label's mask bit must come from position t+1 as well (a label is scored iff the
    TARGET token is assistant content). The last column has no next token: mask 0.
    """
    T = max(item["input_ids"].size(0) for item in batch)
    B = len(batch)
    input_ids = torch.zeros(B, T, dtype=torch.long)
    labels = torch.zeros(B, T, dtype=torch.long)
    loss_mask = torch.zeros(B, T, dtype=torch.float32)
    attention_mask = torch.zeros(B, T, dtype=torch.float32)
    for b, item in enumerate(batch):
        n = item["input_ids"].size(0)
        input_ids[b, :n] = item["input_ids"]
        attention_mask[b, :n] = 1.0
        labels[b, : n - 1] = item["input_ids"][1:]
        loss_mask[b, : n - 1] = item["loss_mask"][1:]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
    }


# -------------------------------------------------------------------------------------
# 2) Preference pairs — for the reward model
# -------------------------------------------------------------------------------------


def _full_dialogue_ids(turns: List[dict], block_size: int) -> List[int]:
    ids = encode(format_chat(turns)) + [EOT_ID]
    return ids[:block_size]


class PreferenceDataset(Dataset):
    """
    One item per HH row: the chosen and rejected dialogues tokenized independently.
    They share their prompt prefix semantically; we do not share it in code —
    clarity over perf.
    """

    def __init__(self, hh_rows: List[dict], block_size: int = 1024):
        self.block_size = block_size
        self.pairs: List[Tuple[List[int], List[int]]] = []
        for row in hh_rows:
            chosen = parse_hh_dialogue(row["chosen"])
            rejected = parse_hh_dialogue(row["rejected"])
            if not chosen or not rejected:
                continue
            c = _full_dialogue_ids(chosen, block_size)
            r = _full_dialogue_ids(rejected, block_size)
            if c == r:
                continue  # identical after truncation: no training signal
            self.pairs.append((c, r))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        c, r = self.pairs[i]
        return {
            "chosen_ids": torch.tensor(c, dtype=torch.long),
            "rejected_ids": torch.tensor(r, dtype=torch.long),
        }


def rm_collate(batch: List[dict]) -> dict:
    """
    Right-pad chosen and rejected to their own max lengths. last_idx is where the
    reward head pools: the index of the last REAL token (= mask.sum() - 1).
    """
    out = {}
    for side in ("chosen", "rejected"):
        seqs = [item[f"{side}_ids"] for item in batch]
        T = max(s.size(0) for s in seqs)
        ids = torch.zeros(len(seqs), T, dtype=torch.long)
        mask = torch.zeros(len(seqs), T, dtype=torch.float32)
        for b, s in enumerate(seqs):
            ids[b, : s.size(0)] = s
            mask[b, : s.size(0)] = 1.0
        out[f"{side}_ids"] = ids
        out[f"{side}_mask"] = mask
        out[f"{side}_last_idx"] = mask.sum(dim=1).long() - 1
    return out


# -------------------------------------------------------------------------------------
# 3) Prompt-only — for PPO rollouts
# -------------------------------------------------------------------------------------


class PromptDataset(Dataset):
    """
    Prompts (dialogue minus the final assistant turn) with the trailing
    "<|im_start|>assistant\\n" cue, ready for generation. Prompts longer than
    prompt_max_len are dropped, not truncated — cutting mid-dialogue breaks the
    chat format.
    """

    def __init__(self, hh_rows: List[dict], prompt_max_len: int = 512):
        self.prompt_max_len = prompt_max_len
        self.prompts: List[List[int]] = []
        seen = set()
        for row in hh_rows:
            turns = parse_hh_dialogue(row["chosen"])
            if not turns or turns[-1]["role"] != "assistant":
                continue
            prompt_turns, _ = split_prompt_and_last_response(turns)
            ids = encode(format_prompt(prompt_turns))
            if len(ids) == 0 or len(ids) > prompt_max_len:
                continue
            key = tuple(ids)
            if key in seen:
                continue
            seen.add(key)
            self.prompts.append(ids)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, i):
        return {"prompt_ids": torch.tensor(self.prompts[i], dtype=torch.long)}


def prompt_collate(batch: List[dict]) -> dict:
    """
    LEFT-pad prompts so every row's last real token sits in the final column and
    generation extends from there. Pad id is EOT (arbitrary; masked anyway).
    """
    T = max(item["prompt_ids"].size(0) for item in batch)
    B = len(batch)
    ids = torch.full((B, T), EOT_ID, dtype=torch.long)
    mask = torch.zeros(B, T, dtype=torch.float32)
    for b, item in enumerate(batch):
        n = item["prompt_ids"].size(0)
        ids[b, T - n:] = item["prompt_ids"]
        mask[b, T - n:] = 1.0
    return {"prompt_ids": ids, "prompt_mask": mask}
