import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader

from data import SupervisedDataset, collate_supervised, build_tokenizer
from gpt import GPT, GPTConfig


def cosine_schedule(step: int, total_steps: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


@dataclass
class SFTConfig:
    data_path: str
    out_dir: str = "weights"
    batch_size: int = 4
    lr: float = 5e-5
    epochs: int = 1
    accum: int = 1
    warmup: int = 100
    init_path: Optional[str] = None
    dropout: float = 0.0
    device: Optional[str] = None


def train_sft(config: SFTConfig) -> None:
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model_config = GPTConfig(dropout=config.dropout)
    model = GPT(model_config).to(device)

    if config.init_path:
        state = torch.load(config.init_path, map_location="cpu")
        model.load_state_dict(state, strict=False)

    dataset = SupervisedDataset(config.data_path, block_size=model_config.block_size)
    bundle = build_tokenizer()
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_supervised(batch, bundle.pad),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.95))

    total_steps = config.epochs * math.ceil(len(loader) / config.accum)
    step = 0

    model.train()
    for epoch in range(config.epochs):
        for batch in loader:
            inputs = batch["input_ids"].to(device)
            targets = batch["target_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            _, loss = model(inputs, targets=targets, attention_mask=mask)
            loss = loss / config.accum
            loss.backward()

            if (step + 1) % config.accum == 0:
                lr = cosine_schedule(step // config.accum, total_steps, config.lr, config.warmup)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            step += 1

        os.makedirs(config.out_dir, exist_ok=True)
        fname = os.path.join(config.out_dir, f"sft_epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), fname)


if __name__ == "__main__":
    RUN = SFTConfig(
        data_path="/path/to/anthropic_hh_sft.jsonl",
        out_dir="weights",
        batch_size=4,
        lr=5e-5,
        epochs=1,
        accum=1,
        warmup=100,
        init_path=None,
        dropout=0.0,
    )
    train_sft(RUN)
