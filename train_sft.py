import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from data import SupervisedDataset, collate_supervised
from gpt import GPT, GPTConfig


def supervised_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross entropy between ``logits`` and ``targets`` masked on padding.

    The mask marks the valid positions inside each sequence.  ``targets`` is the
    input sequence shifted by one position which makes the language modelling
    objective explicit: the model learns to predict the next token ``y`` from the
    observed prefix ``x``.
    """

    vocab = logits.size(-1)
    flat_logits = logits.view(-1, vocab)
    flat_targets = targets.view(-1)
    flat_mask = mask.view(-1).to(logits.dtype)

    per_token = torch.nn.functional.cross_entropy(
        flat_logits,
        flat_targets,
        reduction="none",
        ignore_index=-1,
    )
    denom = flat_mask.sum().clamp(min=1.0)
    return (per_token * flat_mask).sum() / denom


def train_sft(
    data_path: str,
    *,
    out_dir: str = "weights",
    batch_size: int = 4,
    lr: float = 5e-5,
    epochs: int = 1,
    init_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> str:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Supervised data not found at {data_path}. Prepare the JSONL file before training."
        )

    config = GPTConfig()
    model = GPT(config).to(device)

    if init_path:
        state = torch.load(init_path, map_location="cpu")
        model.load_state_dict(state, strict=False)

    dataset = SupervisedDataset(data_path, block_size=config.block_size)
    bundle = dataset.bundle
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_supervised(batch, bundle.pad),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    for _ in range(epochs):
        for batch in loader:
            tokens = batch["input_ids"].to(device)
            targets = batch["target_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            logits, _ = model(tokens, attention_mask=mask)
            loss = supervised_loss(logits, targets, mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "sft.pt")
    torch.save(model.state_dict(), output_path)
    return output_path


def main() -> None:
    data_path = "data/hh_rlhf_sft_train.jsonl"
    out_dir = "weights"
    batch_size = 4
    lr = 5e-5
    epochs = 1
    init_path = None
    device = None

    device_obj = torch.device(device) if device else None
    train_sft(
        data_path,
        out_dir=out_dir,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        init_path=init_path,
        device=device_obj,
    )


if __name__ == "__main__":
    main()
