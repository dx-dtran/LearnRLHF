"""
tests/test_data.py — Module 0.2 / 2.3 / 3.2 / 5.x

Unit tests for dataset wrappers that do NOT require downloading HH. Uses hand-built
fake HH rows so CI can run offline.
"""

import pytest
import torch

from data_hh import (
    parse_hh_dialogue,
    split_prompt_and_last_response,
    SFTDataset,
    PreferenceDataset,
    PromptDataset,
    sft_collate,
    rm_collate,
    prompt_collate,
)


FAKE_CHOSEN = "\n\nHuman: Hi.\n\nAssistant: Hello!\n\nHuman: Another?\n\nAssistant: Sure."
FAKE_REJECTED = "\n\nHuman: Hi.\n\nAssistant: Hello!\n\nHuman: Another?\n\nAssistant: No."
FAKE_ROWS = [
    {"chosen": FAKE_CHOSEN, "rejected": FAKE_REJECTED},
    {"chosen": FAKE_CHOSEN, "rejected": FAKE_REJECTED},
    {"chosen": FAKE_CHOSEN, "rejected": FAKE_REJECTED},
]


# -------------------------------------------------------------------------------------
# Problem 0.2 — parsing
# -------------------------------------------------------------------------------------

def test_parse_hh_dialogue_roles_alternate():
    turns = parse_hh_dialogue(FAKE_CHOSEN)
    roles = [t["role"] for t in turns]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert turns[0]["content"] == "Hi."
    assert turns[-1]["content"] == "Sure."


def test_split_prompt_and_last_response():
    turns = parse_hh_dialogue(FAKE_CHOSEN)
    prompt, last = split_prompt_and_last_response(turns)
    assert len(prompt) == len(turns) - 1
    assert last == "Sure."
    assert prompt[-1]["role"] == "user"


# -------------------------------------------------------------------------------------
# Problem 2.3 — SFT dataset + collate
# -------------------------------------------------------------------------------------

def test_sft_dataset_item_shapes():
    ds = SFTDataset(FAKE_ROWS, block_size=1024)
    assert len(ds) > 0
    item = ds[0]
    assert item["input_ids"].dtype == torch.long
    assert item["loss_mask"].dtype == torch.float32
    assert item["input_ids"].shape == item["loss_mask"].shape


def test_sft_collate_alignment():
    ds = SFTDataset(FAKE_ROWS, block_size=1024)
    batch = sft_collate([ds[0], ds[1]])
    for k in ("input_ids", "labels", "loss_mask", "attention_mask"):
        assert k in batch
    B, T = batch["input_ids"].shape
    assert batch["labels"].shape == (B, T)
    assert batch["loss_mask"].shape == (B, T)
    assert batch["attention_mask"].shape == (B, T)
    # labels[t] should equal input_ids[t+1] on valid positions
    for b in range(B):
        for t in range(T - 1):
            if batch["attention_mask"][b, t + 1]:
                assert batch["labels"][b, t].item() == batch["input_ids"][b, t + 1].item()


# -------------------------------------------------------------------------------------
# Problem 3.2 — preference pair dataset
# -------------------------------------------------------------------------------------

def test_preference_dataset_distinct_tails():
    ds = PreferenceDataset(FAKE_ROWS, block_size=1024)
    item = ds[0]
    c = item["chosen_ids"].tolist()
    r = item["rejected_ids"].tolist()
    assert c != r, "chosen and rejected tokenizations should differ on the last turn"


def test_rm_collate_last_idx_correct():
    ds = PreferenceDataset(FAKE_ROWS, block_size=1024)
    batch = rm_collate([ds[0], ds[1]])
    for side in ("chosen", "rejected"):
        ids = batch[f"{side}_ids"]
        mask = batch[f"{side}_mask"]
        last_idx = batch[f"{side}_last_idx"]
        B = ids.size(0)
        for b in range(B):
            assert last_idx[b].item() == int(mask[b].sum().item()) - 1


# -------------------------------------------------------------------------------------
# Problem 5.x — prompt dataset
# -------------------------------------------------------------------------------------

def test_prompt_dataset_ends_in_assistant_cue():
    from tokenizer import decode
    ds = PromptDataset(FAKE_ROWS, prompt_max_len=512)
    assert len(ds) > 0
    ids = ds[0]["prompt_ids"].tolist()
    text = decode(ids)
    # Prompt should end right before the assistant generates — the cue is open
    assert text.endswith("<|im_start|>assistant\n"), f"prompt ends with: {repr(text[-40:])}"


def test_prompt_collate_left_pads():
    ds = PromptDataset(FAKE_ROWS, prompt_max_len=512)
    batch = prompt_collate([ds[0], ds[1]])
    ids = batch["prompt_ids"]
    mask = batch["prompt_mask"]
    assert ids.shape == mask.shape
    # left-padded: the LAST real token of each row must be at column T-1
    B, T = ids.shape
    for b in range(B):
        assert mask[b, -1].item() == 1, "prompts must be LEFT-padded so generation extends from col T-1"