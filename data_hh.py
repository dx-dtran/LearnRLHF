"""
data_hh.py — Anthropic HH-RLHF download + three derived datasets.

Modules this file covers:
    0.2  inspection (token length stats)
    2.3  SFT DataLoader
    3.2  preference-pair DataLoader
    5.x  prompt-only DataLoader for PPO rollouts

Three datasets produced here:

    1) SFTDataset         -> {"input_ids", "labels", "loss_mask", "attention_mask"}
                              from the *chosen* dialogue, with prompt tokens masked out.
    2) PreferenceDataset  -> {"chosen_ids", "chosen_mask", "chosen_last_idx",
                              "rejected_ids", "rejected_mask", "rejected_last_idx"}
                              for Bradley–Terry RM training.
    3) PromptDataset      -> {"prompt_ids", "prompt_mask"}
                              just the prompts for PPO rollouts (<= prompt_max_len).

The raw HH data is two strings per example, `chosen` and `rejected`, both shaped:

    \\n\\nHuman: ...\\n\\nAssistant: ...\\n\\nHuman: ...\\n\\nAssistant: ...

These are translated into `turns = [{"role": "user", "content": "..."}, ...]` and then
the chat template from `tokenizer.py` is applied. The last "Assistant:" turn is the
one that differs between chosen and rejected.
"""

import json
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import build_sft_example, encode, format_prompt, format_chat


# -------------------------------------------------------------------------------------
# Problem 0.2 — HH download + parsing
# -------------------------------------------------------------------------------------

def parse_hh_dialogue(text: str) -> List[dict]:
    """
    Parse the raw HH dialogue string into a list of turn dicts.

    Input looks like:

        \\n\\nHuman: Hi.\\n\\nAssistant: Hello!\\n\\nHuman: How are you?\\n\\nAssistant: Good.

    Output:
        [{"role":"user","content":"Hi."},
         {"role":"assistant","content":"Hello!"},
         {"role":"user","content":"How are you?"},
         {"role":"assistant","content":"Good."}]

    TODO(0.2): implement. Be tolerant of leading/trailing whitespace and of dialogues
    that start with "Assistant:" (rare but present).
    """
    raise NotImplementedError("TODO(0.2): parse_hh_dialogue")


def download_hh(split: str = "train", cache_dir: str = "hh_cache") -> List[dict]:
    """
    Download (and cache) the Anthropic/hh-rlhf dataset split as a list of
    {"chosen": str, "rejected": str} dicts.

    TODO(0.2): implement with:

        from datasets import load_dataset
        ds = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)

    Then materialize into a python list. (HH is small enough that we can.)
    """
    raise NotImplementedError("TODO(0.2): download_hh")


def split_prompt_and_last_response(dialogue: List[dict]) -> Tuple[List[dict], str]:
    """
    Return (prompt_turns, last_assistant_content) where prompt_turns is everything
    EXCEPT the final assistant message. Used for the RM and for PPO prompts.
    """
    # TODO(0.2): implement. Must assert the last turn is role=assistant.
    raise NotImplementedError("TODO(0.2): split_prompt_and_last_response")


# -------------------------------------------------------------------------------------
# Problem 2.3 — SFT dataset
# -------------------------------------------------------------------------------------

class SFTDataset(Dataset):
    """
    Builds SFT examples from the *chosen* dialogues.

    Each item: (input_ids: list[int], loss_mask: list[int]) from
    tokenizer.build_sft_example. Truncate to `block_size` tokens total (keep the front —
    truncating the tail removes assistant supervision).

    TODO(2.3): implement __init__ (iterate raw HH dicts -> parsed turns ->
    build_sft_example) and __getitem__.
    """

    def __init__(self, hh_rows: List[dict], block_size: int = 1024):
        self.block_size = block_size
        self.examples: list[Tuple[List[int], List[int]]] = []
        # TODO(2.3): populate self.examples. Skip any example whose loss_mask has no 1s
        # after truncation (means we truncated away the assistant turn).
        raise NotImplementedError("TODO(2.3): SFTDataset.__init__")

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
    Right-pad to max length in batch. Return:
        input_ids     : (B, T)  padded with 0 (any token is fine — masked out)
        labels        : (B, T)  = input_ids shifted left by 1, last col pad
        loss_mask     : (B, T)  1 on assistant-content positions of labels
        attention_mask: (B, T)  1 on real tokens in input_ids

    Note on the shift:
        At position t, the model predicts input_ids[t+1]. So labels[t] = input_ids[t+1],
        and loss_mask[t] should correspond to whether the *next* token is an assistant
        content token. Be careful: the mask from build_sft_example is aligned with
        `input_ids`, so you must shift it too.

    TODO(2.3): implement.
    """
    raise NotImplementedError("TODO(2.3): sft_collate")


# -------------------------------------------------------------------------------------
# Problem 3.2 — Preference pair dataset
# -------------------------------------------------------------------------------------

class PreferenceDataset(Dataset):
    """
    Each item is a pair (chosen_full_text, rejected_full_text) tokenized independently.
    The two share their prompt prefix semantically, but tokenizing them separately
    keeps the code simple. The reward model does not need prefix sharing to work.

    Returns dict of python lists; collate handles padding.

    TODO(3.2): implement __init__ to materialize (chosen_ids, rejected_ids) per row.
    Truncate both to `block_size`.
    """

    def __init__(self, hh_rows: List[dict], block_size: int = 1024):
        self.block_size = block_size
        self.pairs: list[Tuple[List[int], List[int]]] = []
        # TODO(3.2): parse each row -> two turn-lists (same prompt, differing final
        # assistant content) -> two full tokenizations -> append.
        raise NotImplementedError("TODO(3.2): PreferenceDataset.__init__")

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
    Pad chosen and rejected to their individual max lengths (they can differ).
    Return:
        chosen_ids, chosen_mask, chosen_last_idx       # (B, Tc), (B, Tc), (B,)
        rejected_ids, rejected_mask, rejected_last_idx # (B, Tr), (B, Tr), (B,)

    last_idx = index of the last REAL token (= mask.sum(-1) - 1). This is where the
    reward head will pool.

    TODO(3.2): implement.
    """
    raise NotImplementedError("TODO(3.2): rm_collate")


# -------------------------------------------------------------------------------------
# Problem 5.x — Prompt-only dataset for PPO rollouts
# -------------------------------------------------------------------------------------

class PromptDataset(Dataset):
    """
    Produces just the prompt (no final assistant response) with the `<|im_start|>assistant\\n`
    cue appended, ready for `model.generate`. Truncate to `prompt_max_len` — if a prompt
    is longer, drop the example (don't truncate mid-dialogue; it breaks the chat format).

    TODO(5.x): implement.
    """

    def __init__(self, hh_rows: List[dict], prompt_max_len: int = 512):
        self.prompt_max_len = prompt_max_len
        self.prompts: list[List[int]] = []
        # TODO(5.x)
        raise NotImplementedError("TODO(5.x): PromptDataset.__init__")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, i):
        return {"prompt_ids": torch.tensor(self.prompts[i], dtype=torch.long)}


def prompt_collate(batch: List[dict]) -> dict:
    """LEFT-pad prompts so generation extends from the rightmost real token."""
    # TODO(5.x): implement. Return prompt_ids (B, T) and prompt_mask (B, T).
    raise NotImplementedError("TODO(5.x): prompt_collate")