import json
import tempfile

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from data import SupervisedDataset, PreferenceDataset, collate_supervised, collate_preferences


def write_jsonl(rows):
    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8")
    for row in rows:
        tmp.write(json.dumps(row) + "\n")
    tmp.flush()
    return tmp


def test_supervised_collate_shapes():
    tmp = write_jsonl([
        {"prompt": "Hello", "chosen": " world"},
        {"prompt": "Another", "chosen": " example"},
    ])
    dataset = SupervisedDataset(tmp.name, block_size=32)
    bundle = dataset.bundle
    batch = [dataset[i] for i in range(len(dataset))]
    collated = collate_supervised(batch, bundle.pad)

    assert collated["input_ids"].ndim == 2
    assert collated["target_ids"].shape == collated["input_ids"].shape
    assert collated["attention_mask"].shape == collated["input_ids"].shape
    assert torch.all(collated["target_ids"][collated["target_ids"] == -1] == -1)


def test_preference_collate_shapes():
    tmp = write_jsonl(
        [
            {
                "prompt": "Hello",
                "chosen": " positive",
                "rejected": " negative",
            }
        ]
    )
    dataset = PreferenceDataset(tmp.name, block_size=32)
    bundle = dataset.bundle
    batch = [dataset[0]]
    collated = collate_preferences(batch, bundle.pad)

    assert collated["prompt"].ndim == 2
    assert collated["chosen"].shape == collated["chosen_mask"].shape
    assert collated["rejected"].shape == collated["rejected_mask"].shape
