"""
train_rm.py — reward model on HH preference pairs.

Part 3 of the crash course. Your job:

    [FILL 3.1]  RewardModel.forward — pool the per-token scores at the last real token
    [FILL 3.2]  bt_loss — Bradley–Terry pairwise loss

The training loop is provided. Run with:

    python train_rm.py        (expects sft.pt from Part 2)

Math you should be able to reproduce on paper (see notes/03-rm.md):
    Bradley–Terry says P(chosen beats rejected) = sigmoid(r_c - r_r). The negative
    log-likelihood of one pair is

        L = -log sigmoid(r_c - r_r) = softplus(r_r - r_c)

    with gradients

        dL/dr_c = sigmoid(r_c - r_r) - 1     (always negative: pushes r_c up)
        dL/dr_r = 1 - sigmoid(r_c - r_r)     (always positive: pushes r_r down)

    They sum to zero, so the loss is invariant to adding a constant to every score —
    only differences are learned.
"""

import os as _os
import sys as _sys

# make `python solutions/train_*.py` runnable: repo root provides data_hh/config,
# while this directory's own model/tokenizer/ppo_core shadow the scaffolds
_sys.path.insert(1, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig, RMConfig
from model import GPT, ScalarHead


# =====================================================================================
# [FILL 3.1] — reward model = GPT backbone + scalar head, pooled at last real token
# =====================================================================================


class RewardModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.backbone = GPT(config)
        self.reward_head = ScalarHead(config.n_embd)

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, T)
        attention_mask: torch.Tensor,  # (B, T)
        last_idx: torch.Tensor,        # (B,) long — index of last real token
    ) -> torch.Tensor:
        """
        One scalar reward per sequence.

            hidden = backbone.forward_hidden(...)      # (B, T, C)
            scores = reward_head(hidden)               # (B, T)
            reward = scores at column last_idx[b] for each row b   # (B,)

        Pooling at the LAST REAL token matters: with right padding, column T-1 is
        usually padding garbage. Use gather, not [:, -1].
        """
        hidden = self.backbone.forward_hidden(input_ids, attention_mask)
        scores = self.reward_head(hidden)
        return scores.gather(1, last_idx.unsqueeze(1)).squeeze(1)


# =====================================================================================
# [FILL 3.2] — Bradley–Terry pairwise loss
# =====================================================================================


def bt_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """
    Mean preference loss over a batch of pairs:

        per_pair = softplus(r_rejected - r_chosen)     # = -log sigmoid(r_c - r_r)
        return per_pair.mean()

    softplus, not -log(sigmoid(.)): same function, but it doesn't overflow when the
    score gap is large.
    """
    return F.softplus(r_rejected - r_chosen).mean()


def pairwise_accuracy(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """Fraction of pairs ranked correctly."""
    return (r_chosen > r_rejected).float().mean()


# =====================================================================================
# Training loop (provided)
# =====================================================================================


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int) -> tuple:
    from train_sft import autocast_ctx

    model.eval()
    losses, accs = [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast_ctx(device):
            r_c = model(batch["chosen_ids"], batch["chosen_mask"], batch["chosen_last_idx"])
            r_r = model(batch["rejected_ids"], batch["rejected_mask"], batch["rejected_last_idx"])
        losses.append(bt_loss(r_c, r_r).item())
        accs.append(pairwise_accuracy(r_c, r_r).item())
    model.train()
    n = max(len(losses), 1)
    return sum(losses) / n, sum(accs) / n


def train_rm():
    from torch.utils.data import DataLoader

    from data_hh import PreferenceDataset, download_hh, rm_collate
    from train_sft import autocast_ctx, build_optimizer, cosine_lr

    cfg = RMConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RewardModel(GPTConfig())
    ckpt = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
    model.backbone.load_state_dict(ckpt["model"])
    model.to(device)
    model.train()

    train_ds = PreferenceDataset(download_hh("train"), block_size=cfg.block_size)
    eval_ds = PreferenceDataset(download_hh("test"), block_size=cfg.block_size)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=rm_collate
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=rm_collate
    )

    optim = build_optimizer(model, cfg.lr, cfg.weight_decay, cfg.betas)
    total_steps = cfg.epochs * len(train_loader) // cfg.accum_steps
    print(f"{len(train_ds)} pairs, {total_steps} optimizer steps on {device}")

    opt_step = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast_ctx(device):
                r_c = model(batch["chosen_ids"], batch["chosen_mask"], batch["chosen_last_idx"])
                r_r = model(batch["rejected_ids"], batch["rejected_mask"], batch["rejected_last_idx"])
                loss = bt_loss(r_c, r_r)
            (loss / cfg.accum_steps).backward()

            if (step + 1) % cfg.accum_steps == 0:
                # linear warmup, flat after (single epoch; cosine buys little)
                lr = cfg.lr * min((opt_step + 1) / max(cfg.warmup_steps, 1), 1.0)
                for g in optim.param_groups:
                    g["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()
                optim.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % cfg.log_every == 0:
                    acc = pairwise_accuracy(r_c, r_r).item()
                    print(
                        f"step {opt_step}/{total_steps} loss {loss.item():.4f} "
                        f"acc {acc:.3f} ({time.time() - t0:.0f}s)"
                    )
                if opt_step % cfg.eval_every == 0:
                    ev_loss, ev_acc = evaluate(model, eval_loader, device, cfg.eval_batches)
                    print(f"eval loss {ev_loss:.4f} pairwise acc {ev_acc:.3f}")

    torch.save({"model": model.state_dict(), "config": GPTConfig()}, cfg.save_path)
    print(f"saved {cfg.save_path}")


if __name__ == "__main__":
    train_rm()
