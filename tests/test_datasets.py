import json
import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from data import PreferenceDataset, SupervisedDataset


def write_jsonl(rows):
    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8")
    for row in rows:
        tmp.write(json.dumps(row) + "\n")
    tmp.flush()
    return tmp


def test_supervised_dataset_provides_ready_tensors():
    tmp = write_jsonl([
        {"prompt": "Hello", "chosen": " world"},
        {"prompt": "Another", "chosen": " example"},
    ])
    dataset = SupervisedDataset(tmp.name, block_size=32)

    sample = dataset[0]
    assert sample["input_ids"].shape == (dataset.block_size,)
    assert sample["target_ids"].shape == (dataset.block_size,)
    assert sample["attention_mask"].sum() > 0
    assert torch.all(sample["target_ids"][sample["attention_mask"] == 0] == -1)


def test_preference_dataset_includes_masks():
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

    sample = dataset[0]
    for key in ("prompt", "chosen", "rejected"):
        tokens = sample[key]
        mask = sample[f"{key}_mask"]
        assert tokens.shape == (dataset.block_size,)
        assert mask.shape == tokens.shape
        assert mask.sum() > 0
