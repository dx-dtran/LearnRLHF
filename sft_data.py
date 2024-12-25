import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class SupervisedFineTuningDataset(Dataset):
    def __init__(self, jsonl_file, block_size=1024):
        self.examples = []
        self.block_size = block_size

        with open(jsonl_file, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                input_ids = data["chosen"]

                if len(input_ids) > block_size:
                    input_ids = input_ids[:block_size]

                target_ids = input_ids[1:] + [-100]

                self.examples.append(
                    {
                        "input_ids": input_ids,
                        "target_ids": target_ids,
                    }
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_ids = example["input_ids"]
        target_ids = example["target_ids"]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }


def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    target_ids = [item["target_ids"] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids_padded,
        "target_ids": target_ids_padded,
    }


if __name__ == "__main__":
    dataset = SupervisedFineTuningDataset("train.jsonl", block_size=1024)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    for batch in dataloader:
        input_ids = batch["input_ids"]
        target_ids = batch["target_ids"]

        print("Input IDs:", input_ids.shape)
        print("Target IDs:", target_ids.shape)
        break
