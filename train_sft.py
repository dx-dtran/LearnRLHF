import math
import os
import time
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import SupervisedDataset, collate_supervised, build_tokenizer
from gpt import GPT, GPTConfig
from simple_logger import TrainingLogger


def cosine_schedule(step: int, total_steps: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def supervised_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the masked language modeling loss used for SFT training."""

    log_probs = F.log_softmax(logits, dim=-1)
    loss = F.nll_loss(
        log_probs.transpose(1, 2),
        targets,
        reduction="none",
        ignore_index=-1,
    )
    mask = mask.to(log_probs.dtype)
    denom = mask.sum().clamp(min=1.0)
    return (loss * mask).sum() / denom


def train_sft(
    data_path: str,
    *,
    out_dir: str = "weights",
    batch_size: int = 4,
    lr: float = 5e-5,
    epochs: int = 1,
    accum: int = 1,
    warmup: int = 100,
    init_path: Optional[str] = None,
    dropout: float = 0.0,
    device: Optional[torch.device] = None,
) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = GPTConfig(dropout=dropout)
    model = GPT(config).to(device)

    if init_path:
        state = torch.load(init_path, map_location="cpu")
        model.load_state_dict(state, strict=False)

    dataset = SupervisedDataset(data_path, block_size=config.block_size)
    bundle = build_tokenizer()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_supervised(batch, bundle.pad),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    updates_per_epoch = math.ceil(len(loader) / accum)
    total_updates = epochs * updates_per_epoch
    update_index = 0
    micro_step = 0
    update_timer = time.perf_counter()
    accumulated_loss = 0.0

    logger = TrainingLogger("sft")

    model.train()
    try:
        for epoch in range(epochs):
            epoch_updates = 0
            for batch in loader:
                micro_step += 1
                if (micro_step - 1) % accum == 0:
                    update_timer = time.perf_counter()
                    accumulated_loss = 0.0

                inputs = batch["input_ids"].to(device)
                targets = batch["target_ids"].to(device)
                mask = batch["attention_mask"].to(device).to(inputs.dtype)

                logits, _ = model(inputs, attention_mask=mask)
                loss = supervised_loss(logits, targets, mask) / accum
                accumulated_loss += loss.item()
                loss.backward()

                performed_step = False
                if micro_step % accum == 0:
                    lr_now = cosine_schedule(update_index, total_updates, lr, warmup)
                    for g in optimizer.param_groups:
                        g["lr"] = lr_now
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    update_index += 1
                    epoch_updates += 1
                    performed_step = True

                if performed_step:
                    elapsed = time.perf_counter() - update_timer
                    logger.log(
                        {
                            "epoch": epoch + 1,
                            "epoch_iteration": epoch_updates,
                            "iteration": update_index,
                            "loss": accumulated_loss,
                            "lr": optimizer.param_groups[0]["lr"],
                            "step_time": elapsed,
                        }
                    )
                    update_timer = time.perf_counter()
                    accumulated_loss = 0.0

            if micro_step % accum != 0:
                lr_now = cosine_schedule(update_index, total_updates, lr, warmup)
                for g in optimizer.param_groups:
                    g["lr"] = lr_now
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                update_index += 1
                epoch_updates += 1
                elapsed = time.perf_counter() - update_timer
                logger.log(
                    {
                        "epoch": epoch + 1,
                        "epoch_iteration": epoch_updates,
                        "iteration": update_index,
                        "loss": accumulated_loss,
                        "lr": optimizer.param_groups[0]["lr"],
                        "step_time": elapsed,
                    }
                )
                update_timer = time.perf_counter()
                accumulated_loss = 0.0

            micro_step = update_index * accum

        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, f"sft_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), fname)
    finally:
        logger.close()


if __name__ == "__main__":
    DATA_PATH = "data/sft_train.jsonl"
    OUT_DIR = "weights"
    BATCH_SIZE = 4
    LEARNING_RATE = 5e-5
    EPOCHS = 1
    ACCUM = 1
    WARMUP = 100
    INIT_PATH = None
    DROPOUT = 0.0

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Supervised data not found at {DATA_PATH}. Update the path before running."
        )

    train_sft(
        DATA_PATH,
        out_dir=OUT_DIR,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        accum=ACCUM,
        warmup=WARMUP,
        init_path=INIT_PATH,
        dropout=DROPOUT,
    )
