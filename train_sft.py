"""
train_sft.py — Supervised Fine-Tuning.

Modules this file covers:
    2.2  Masked causal-LM loss
    2.4  SFT training loop
    2.5  (invoked via eval.py)

The InstructGPT-style SFT objective: standard next-token cross-entropy, but only on
assistant-content tokens. Zero gradient through user/system/scaffold tokens.

Derivation (put in notes/02-sft.md before implementing):
    For a single example of length T with mask m in {0,1}^T:
        L = - (1/N) Σ_t  m_t · log p_θ(y_t | x_<t),       N = Σ_t m_t
    Let ℓ_t = CE per position. Then ∂L/∂logits_t = m_t · (softmax(logits_t) - onehot(y_t)) / N.
    Averaging is token-level across the batch (the usual choice), not per example.
    Token-level averaging weights long responses more.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================================
# Problem 2.2 — Masked SFT loss
# =====================================================================================

def sft_loss(
    logits: torch.Tensor,   # (B, T, V)
    labels: torch.Tensor,   # (B, T)   long
    loss_mask: torch.Tensor,  # (B, T) float, 1 on positions we score
) -> torch.Tensor:
    """
    Mean next-token cross-entropy over positions where loss_mask == 1.

    Implementation:
        logp = F.log_softmax(logits, dim=-1)                  # (B,T,V)
        nll  = -logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B,T)
        loss = (nll * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
        return loss

    Do NOT use `F.cross_entropy(..., ignore_index=-100)`. The mask is multiplied in
    explicitly so the gradient path stays visible.

    TODO(2.2): implement.

    Your gradient test (tests/test_grad_sft.py::test_sft_loss_grad):
        - fp64 tiny logits
        - autograd vs centered-difference; rel err < 1e-5
        - flipping a masked-out label must NOT change the loss
    """
    raise NotImplementedError("TODO(2.2): sft_loss")


# =====================================================================================
# Problem 2.4 — SFT training loop (skeleton)
# =====================================================================================

def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple,
) -> torch.optim.Optimizer:
    """
    Build AdamW with the nanoGPT convention: weight-decay only on 2D parameters.
    No decay on LayerNorm weights, biases, or embedding weights. (GPT-2 often
    keeps decay on embeddings; this code excludes them to match nanoGPT.)

    TODO(2.4): split model.parameters() into two groups:
        decay  = [p for p in params if p.requires_grad and p.dim() >= 2]
        nodecay = the rest
    and build AdamW with those two groups.
    """
    raise NotImplementedError("TODO(2.4): build_optimizer")


def cosine_lr(step: int, warmup: int, total: int, peak: float, min_ratio: float) -> float:
    """
    Linear warmup to `peak` over `warmup` steps, then cosine decay to `peak * min_ratio`
    over the remaining `total - warmup` steps.

    TODO(2.4): implement. Should be pure math, no state.
    """
    raise NotImplementedError("TODO(2.4): cosine_lr")


def train_sft():
    """
    Full loop:
        - load SFTConfig from config.py
        - instantiate GPT and load HF weights
        - build SFTDataset + DataLoader
        - build optimizer + scheduler
        - for epoch in epochs:
            for step, batch in enumerate(loader):
                with torch.autocast(..., dtype=torch.bfloat16):
                    logits = model(batch["input_ids"], batch["attention_mask"])
                    loss = sft_loss(logits[:, :-1], batch["labels"][:, :-1], batch["loss_mask"][:, :-1])
                (loss / accum).backward()
                if (step+1) % accum == 0:
                    clip_grad_norm_(1.0); optim.step(); optim.zero_grad()
                    update lr each micro-step or each optim-step (pick one — document)
                if step % eval_every == 0: run eval on a fixed subset of the held-out split
        - save to sft.pt

    Note on the shift in the loss call: the loader already returns labels as
    input_ids[t+1]. By convention, logits are computed over positions [0..T-1] and
    labels over [0..T-1] aligned (both shifted), hence the `[:, :-1]` slice on logits
    when labels are already shifted. If the collate does the shift inside, drop the
    slice and document the choice with a comment.

    TODO(2.4): implement. A few dozen lines suffice.
    """
    raise NotImplementedError("TODO(2.4): train_sft")


if __name__ == "__main__":
    train_sft()