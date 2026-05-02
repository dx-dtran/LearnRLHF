"""
train_rm.py — Reward model (Bradley–Terry pairwise preference).

Modules this file covers:
    3.1  RM architecture
    3.3  Bradley–Terry loss
    3.4  RM training loop
    3.5  Calibration sanity

Math (put in notes/03-rm.md before you implement):
    Given a preferred completion c and a rejected completion r with scalar rewards
    r_c, r_r, the Bradley–Terry model says
        P(c > r | x)  =  σ(r_c - r_r).
    Maximum-likelihood under this model gives per-pair loss:
        L  =  -log σ(r_c - r_r)  =  softplus(r_r - r_c).
    Derivatives:
        ∂L/∂r_c  =  σ(r_c - r_r) - 1   = -(1 - σ(r_c - r_r))   (negative: pushes r_c UP)
        ∂L/∂r_r  =  1 - σ(r_c - r_r)                            (positive: pushes r_r DOWN)
    The two gradients are exact negatives, so they sum to zero. Bradley–Terry is
    invariant under a constant shift of both scores.

Use `softplus` rather than `-log(sigmoid(.))` to avoid under/overflow for large |Δr|.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig, RMConfig
from model import GPT, ScalarHead


# =====================================================================================
# Problem 3.1 — Reward model architecture
# =====================================================================================

class RewardModel(nn.Module):
    """
    GPT-2 backbone + scalar reward head.

    Forward:
        hidden = backbone.forward_hidden(input_ids, attention_mask)   # (B, T, C)
        scores = reward_head(hidden)                                   # (B, T)
        reward = scores.gather(1, last_idx.unsqueeze(1)).squeeze(1)   # (B,)

    TODO(3.1): implement. Initialize from `sft.pt` in training script (not here).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        # TODO(3.1)
        raise NotImplementedError("TODO(3.1): RewardModel.__init__")

    def forward(
        self,
        input_ids: torch.Tensor,      # (B, T)
        attention_mask: torch.Tensor,  # (B, T)
        last_idx: torch.Tensor,        # (B,) long — position of last real token
    ) -> torch.Tensor:
        """Return (B,) scalar reward per sequence."""
        # TODO(3.1)
        raise NotImplementedError("TODO(3.1): RewardModel.forward")


# =====================================================================================
# Problem 3.3 — Bradley–Terry pairwise loss
# =====================================================================================

def bt_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """
    Mean pairwise preference loss over a batch.

        per_pair = softplus(r_rejected - r_chosen)
        return per_pair.mean()

    TODO(3.3): implement.

    Gradient test (tests/test_grad_rm.py):
        - fp64 random r_c, r_r
        - autograd vs centered-difference
        - also check: if you add a constant C to both r_c and r_r, loss is unchanged
          (BT is translation-invariant in the score).
    """
    raise NotImplementedError("TODO(3.3): bt_loss")


def pairwise_accuracy(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """Fraction of pairs with r_chosen > r_rejected."""
    return (r_chosen > r_rejected).float().mean()


# =====================================================================================
# Problem 3.4 — RM training loop
# =====================================================================================

def train_rm():
    """
    Full loop (sketch):
        - load RMConfig
        - load sft.pt into a new RewardModel (strict=False — the head is new)
        - PreferenceDataset + DataLoader with rm_collate
        - AdamW (reuse train_sft.build_optimizer) + linear warmup (no cosine — 1 epoch)
        - bf16 autocast
        - per step:
            r_c = model(chosen_ids, chosen_mask, chosen_last_idx)
            r_r = model(rejected_ids, rejected_mask, rejected_last_idx)
            loss = bt_loss(r_c, r_r)
            (loss / accum).backward()
            grad_clip, step, zero_grad
            log loss + pairwise_accuracy
        - periodically eval on held-out pairs
        - save to rm.pt

    TODO(3.4): implement.
    """
    raise NotImplementedError("TODO(3.4): train_rm")


if __name__ == "__main__":
    train_rm()