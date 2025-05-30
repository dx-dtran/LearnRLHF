import json
import torch
from torch.utils.data import Dataset, DataLoader


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

    max_length = max(len(seq) for seq in input_ids)

    input_ids_padded = torch.full((len(input_ids), max_length), 0, dtype=torch.long)
    target_ids_padded = torch.full(
        (len(target_ids), max_length), -100, dtype=torch.long
    )
    attention_mask = torch.zeros((len(input_ids), max_length), dtype=torch.bool)

    for i, (input_seq, target_seq) in enumerate(zip(input_ids, target_ids)):
        input_ids_padded[i, : len(input_seq)] = input_seq
        target_ids_padded[i, : len(target_seq)] = target_seq
        attention_mask[i, : len(input_seq)] = True

    return {
        "input_ids": input_ids_padded,
        "target_ids": target_ids_padded,
        "attention_mask": attention_mask,
    }


if __name__ == "__main__":
    dataset = SupervisedFineTuningDataset("train.jsonl", block_size=1024)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    total_tokens = 0
    with open("train.jsonl", "r") as f:
        for line in f:
            data = json.loads(line.strip())
            input_ids = data["chosen"]
            total_tokens += len(input_ids)

    print(f"Total number of tokens in 'chosen' field: {total_tokens}")

    for batch in dataloader:
        input_ids = batch["input_ids"]
        target_ids = batch["target_ids"]
        attention_mask = batch["attention_mask"]

        print("Input IDs:", input_ids)
        print("Target IDs:", target_ids)
        print("Attention Mask:", attention_mask)
        break
