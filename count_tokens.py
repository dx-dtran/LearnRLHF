import json
import torch
from torch.utils.data import DataLoader
from sft_data import SupervisedFineTuningDataset, collate_fn


def calculate_tokens_seen(jsonl_file, block_size, batch_size, num_epochs):
    # Load the dataset
    dataset = SupervisedFineTuningDataset(jsonl_file, block_size=block_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    total_tokens_seen = 0

    # Iterate over the dataset for the specified number of epochs
    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"]

            # Account for padding
            max_length = input_ids.size(1)  # Max sequence length in the batch
            total_tokens_seen += max_length * input_ids.size(
                0
            )  # Tokens = seq_len * batch_size

    return total_tokens_seen


if __name__ == "__main__":
    # Configuration parameters
    train_file = "train.jsonl"
    block_size = 1024  # Maximum sequence length per example
    batch_size = 8  # Number of examples per batch
    num_epochs = 3  # Number of training epochs

    total_tokens = calculate_tokens_seen(
        train_file, block_size=block_size, batch_size=batch_size, num_epochs=num_epochs
    )

    print(f"Total tokens seen during training: {total_tokens}")
