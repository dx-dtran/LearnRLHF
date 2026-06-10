"""
train_sft.py — supervised fine-tuning on the chosen HH dialogues.

Part 2 of the crash course. Your job:

    [FILL 2.2]  sft_loss — masked next-token cross-entropy

The optimizer, schedule, and training loop are provided. Run with:

    python train_sft.py

Derivation you should be able to do on paper (see notes/02-sft.md):
    For one example of length T with mask m in {0,1}^T,

        L = -(1/N) sum_t m_t * log p(y_t | x_<t),     N = sum_t m_t

    and the gradient through the logits at position t is

        dL/dlogits_t = m_t * (softmax(logits_t) - onehot(y_t)) / N.
"""

import os as _os
import sys as _sys

# make `python solutions/train_*.py` runnable: repo root provides data_hh/config,
# while this directory's own model/tokenizer/ppo_core shadow the scaffolds
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig, SFTConfig


# =====================================================================================
# [FILL 2.2] — masked SFT loss
# =====================================================================================


def sft_loss(
    logits: torch.Tensor,     # (B, T, V)
    labels: torch.Tensor,     # (B, T) long
    loss_mask: torch.Tensor,  # (B, T) float, 1 on positions we score
) -> torch.Tensor:
    """
    Mean next-token cross-entropy over positions where loss_mask == 1.

        logp = log_softmax(logits, dim=-1)
        nll  = -logp gathered at the label ids                     # (B, T)
        loss = (nll * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)

    No F.cross_entropy(..., ignore_index=-100) tricks: multiply the mask in
    explicitly so the gradient path stays visible.
    """
    logp = F.log_softmax(logits, dim=-1)
    nll = -logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (nll * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)


# =====================================================================================
# Optimizer + schedule (provided)
# =====================================================================================


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple,
) -> torch.optim.Optimizer:
    """
    AdamW with the nanoGPT convention: weight decay only on parameters with dim >= 2
    (matmul weights, embeddings); none on LayerNorm weights and biases.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    nodecay = [p for p in params if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def cosine_lr(step: int, warmup: int, total: int, peak: float, min_ratio: float) -> float:
    """Linear warmup to `peak`, then cosine decay to `peak * min_ratio`."""
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(progress, 1.0)
    floor = peak * min_ratio
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


def autocast_ctx(device: torch.device):
    """bf16 autocast on CUDA; no-op on CPU (keeps the same code runnable anywhere)."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    import contextlib

    return contextlib.nullcontext()


# =====================================================================================
# Training loop (provided)
# =====================================================================================


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int) -> float:
    model.eval()
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast_ctx(device):
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = sft_loss(logits, batch["labels"], batch["loss_mask"])
        total += loss.item()
        count += 1
    model.train()
    return total / max(count, 1)


def train_sft():
    from torch.utils.data import DataLoader

    from data_hh import SFTDataset, download_hh, sft_collate
    from model import GPT, load_gpt2_from_hf

    cfg = SFTConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT(GPTConfig())
    load_gpt2_from_hf(model, "gpt2")
    model.to(device)
    model.train()

    train_ds = SFTDataset(download_hh("train"), block_size=cfg.block_size)
    eval_ds = SFTDataset(download_hh("test"), block_size=cfg.block_size)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=sft_collate
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=sft_collate
    )

    optim = build_optimizer(model, cfg.lr, cfg.weight_decay, cfg.betas)
    total_steps = cfg.epochs * len(train_loader) // cfg.accum_steps

    print(f"{len(train_ds)} train examples, {total_steps} optimizer steps on {device}")

    opt_step = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast_ctx(device):
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = sft_loss(logits, batch["labels"], batch["loss_mask"])
            (loss / cfg.accum_steps).backward()

            if (step + 1) % cfg.accum_steps == 0:
                lr = cosine_lr(
                    opt_step, cfg.warmup_steps, total_steps, cfg.lr, cfg.min_lr_ratio
                )
                for g in optim.param_groups:
                    g["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()
                optim.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % cfg.log_every == 0:
                    print(
                        f"epoch {epoch} step {opt_step}/{total_steps} "
                        f"loss {loss.item():.4f} lr {lr:.2e} "
                        f"({time.time() - t0:.0f}s)"
                    )
                if opt_step % cfg.eval_every == 0:
                    ev = evaluate(model, eval_loader, device, cfg.eval_batches)
                    print(f"eval loss {ev:.4f}")

    torch.save({"model": model.state_dict(), "config": model.config}, cfg.save_path)
    print(f"saved {cfg.save_path}")


if __name__ == "__main__":
    train_sft()
