"""
train_ppo.py — the PPO loop. Fully provided; you build its internals in ppo_core.py.

Run with:

    python train_ppo.py       (expects sft.pt and rm.pt)

Four models in memory (24GB budget, see notes/05-ppo.md):

    policy       trainable, init from sft.pt
    value head   trainable, ScalarHead on the POLICY's hidden states (shared
                 backbone: cheaper; a separate value backbone is more stable if you
                 see value-loss blowups)
    reference    frozen copy of sft.pt — the KL anchor
    reward model frozen, from rm.pt

Per iteration: rollout (no grad) -> K epochs of minibatch PPO (grad) -> log.

What healthy training looks like:
    mean RM reward up, KL (k3) drifting up SLOWLY, entropy down slowly, clip
    fraction in the 0.05-0.3 range. Reward spiking while KL explodes is reward
    hacking: raise kl_coef or lower policy_lr.
"""

import csv
import os
import time
from collections import defaultdict

import torch

from config import GPTConfig, PPOConfig
from model import GPT, ScalarHead
from ppo_core import (
    gae,
    gather_logprobs,
    generate_with_logprobs,
    kl_k1,
    kl_k3,
    masked_entropy,
    normalize_advantages,
    ppo_policy_loss,
    shape_reward,
    value_loss,
)
from train_rm import RewardModel
from train_sft import autocast_ctx


# =====================================================================================
# Model layout
# =====================================================================================


def build_models(model_cfg: GPTConfig, cfg: PPOConfig, device: torch.device):
    def load_backbone(path: str) -> dict:
        return torch.load(path, map_location="cpu", weights_only=False)["model"]

    policy = GPT(model_cfg)
    policy.load_state_dict(load_backbone(cfg.policy_init))
    policy.to(device).train()

    value_head = ScalarHead(model_cfg.n_embd).to(device).train()

    ref = GPT(model_cfg)
    ref.load_state_dict(load_backbone(cfg.ref_init))
    ref.to(device).eval().requires_grad_(False)

    rm = RewardModel(model_cfg)
    rm.load_state_dict(load_backbone(cfg.rm_init))
    rm.to(device).eval().requires_grad_(False)

    n = sum(p.numel() for p in policy.parameters())
    print(f"policy params: {n/1e6:.0f}M; 4 models on {device}")
    return policy, value_head, ref, rm


# =====================================================================================
# Phase 1: rollout (everything no-grad)
# =====================================================================================


@torch.no_grad()
def rollout(policy, value_head, ref, rm, prompt_ids, prompt_mask, cfg: PPOConfig, eos_token_id):
    device = prompt_ids.device
    T_p = prompt_ids.size(1)
    policy.eval()
    with autocast_ctx(device):
        full_ids, response_ids, logprobs_old, values_old, response_mask = (
            generate_with_logprobs(
                policy,
                value_head,
                prompt_ids,
                prompt_mask,
                cfg.response_max_len,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                eos_token_id=eos_token_id,
            )
        )
        full_mask = torch.cat([prompt_mask.float(), response_mask], dim=1)

        ref_logits = ref(full_ids, attention_mask=full_mask)
        ref_logprobs = gather_logprobs(ref_logits[:, T_p - 1:-1, :], response_ids)

        last_idx = T_p + response_mask.sum(dim=1).long() - 1
        rm_reward = rm(full_ids, full_mask, last_idx)
    policy.train()

    # fp32 for the RL arithmetic
    logprobs_old, ref_logprobs, values_old, response_mask, rm_reward = (
        x.float() for x in (logprobs_old, ref_logprobs, values_old, response_mask, rm_reward)
    )

    kl = kl_k1(logprobs_old, ref_logprobs)
    rewards = shape_reward(rm_reward, kl, response_mask, cfg.kl_coef)
    advantages, returns = gae(rewards, values_old, response_mask, cfg.gamma, cfg.gae_lambda)
    advantages = normalize_advantages(advantages, response_mask)

    n_tok = response_mask.sum().clamp_min(1.0)
    return {
        "full_ids": full_ids,
        "full_mask": full_mask,
        "response_ids": response_ids,
        "response_mask": response_mask,
        "logprobs_old": logprobs_old,
        "values_old": values_old,
        "advantages": advantages,
        "returns": returns,
        "T_p": T_p,
        "stats": {
            "rm_reward": rm_reward.mean().item(),
            "kl_k3": (kl_k3(logprobs_old, ref_logprobs) * response_mask).sum().item()
            / n_tok.item(),
            "response_len": (response_mask.sum(dim=1)).float().mean().item(),
        },
    }


# =====================================================================================
# Phase 2: optimize (K epochs of minibatch PPO)
# =====================================================================================


def optimize(policy, value_head, optimizer, ro: dict, cfg: PPOConfig):
    device = ro["full_ids"].device
    B = ro["full_ids"].size(0)
    T_p = ro["T_p"]
    stats = defaultdict(list)

    for _ in range(cfg.ppo_epochs):
        perm = torch.randperm(B, device=device)
        for start in range(0, B, cfg.minibatch_size):
            idx = perm[start : start + cfg.minibatch_size]
            mb = {k: v[idx] for k, v in ro.items() if torch.is_tensor(v)}

            with autocast_ctx(device):
                hidden = policy.forward_hidden(mb["full_ids"], mb["full_mask"])
                hidden_resp = hidden[:, T_p - 1:-1, :]
                logits_resp = hidden_resp @ policy.wte.weight.t()
                logprobs_new = gather_logprobs(logits_resp, mb["response_ids"])
                values_new = value_head(hidden_resp)

                mask = mb["response_mask"]
                l_pi = ppo_policy_loss(
                    logprobs_new.float(), mb["logprobs_old"], mb["advantages"],
                    mask, cfg.clip_eps,
                )
                l_v = value_loss(
                    values_new.float(), mb["values_old"], mb["returns"],
                    mask, cfg.value_clip_eps,
                )
                h = masked_entropy(logits_resp.float(), mask)
                loss = l_pi + cfg.value_coef * l_v - cfg.entropy_coef * h

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            params = [p for g in optimizer.param_groups for p in g["params"]]
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()

            with torch.no_grad():
                ratio = torch.exp(logprobs_new.float() - mb["logprobs_old"])
                clipped = ((ratio - 1.0).abs() > cfg.clip_eps).float()
                clip_frac = (clipped * mask).sum() / mask.sum().clamp_min(1.0)

            stats["policy_loss"].append(l_pi.item())
            stats["value_loss"].append(l_v.item())
            stats["entropy"].append(h.item())
            stats["clip_frac"].append(clip_frac.item())
            stats["grad_norm"].append(float(grad_norm))

    return {k: sum(v) / len(v) for k, v in stats.items()}


# =====================================================================================
# Logging
# =====================================================================================


CSV_FIELDS = [
    "iter", "rm_reward", "kl_k3", "response_len", "policy_loss", "value_loss",
    "entropy", "clip_frac", "grad_norm", "tokens_per_sec",
]


def log_csv(path: str, row: dict):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def plot_csv(path: str, out_png: str = "ppo_plots.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(path)))
    if len(rows) < 2:
        return
    keys = ["rm_reward", "kl_k3", "policy_loss", "value_loss", "entropy", "clip_frac"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    xs = [int(r["iter"]) for r in rows]
    for ax, k in zip(axes.flat, keys):
        ax.plot(xs, [float(r[k]) for r in rows])
        ax.set_title(k)
        ax.set_xlabel("iter")
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


# =====================================================================================
# Main
# =====================================================================================


def train_ppo():
    from torch.utils.data import DataLoader

    from data_hh import PromptDataset, download_hh, prompt_collate
    from tokenizer import EOT_ID

    cfg = PPOConfig()
    model_cfg = GPTConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy, value_head, ref, rm = build_models(model_cfg, cfg, device)

    prompts = PromptDataset(download_hh("train"), prompt_max_len=cfg.prompt_max_len)
    loader = DataLoader(
        prompts, batch_size=cfg.rollout_batch_size, shuffle=True,
        collate_fn=prompt_collate, drop_last=True,
    )
    print(f"{len(prompts)} prompts")

    optimizer = torch.optim.AdamW(
        [
            {"params": policy.parameters(), "lr": cfg.policy_lr},
            {"params": value_head.parameters(), "lr": cfg.value_lr},
        ],
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )

    it = 0
    batches = iter(loader)
    while it < cfg.num_iters:
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)

        t0 = time.time()
        prompt_ids = batch["prompt_ids"].to(device)
        prompt_mask = batch["prompt_mask"].to(device)

        ro = rollout(policy, value_head, ref, rm, prompt_ids, prompt_mask, cfg, EOT_ID)
        rollout_stats = ro.pop("stats")
        opt_stats = optimize(policy, value_head, optimizer, ro, cfg)

        dt = time.time() - t0
        n_tokens = ro["response_mask"].sum().item()
        row = {
            "iter": it,
            **rollout_stats,
            **opt_stats,
            "tokens_per_sec": n_tokens / dt,
        }
        log_csv(cfg.log_csv, row)
        if it % cfg.log_every == 0:
            print(
                f"iter {it} reward {row['rm_reward']:.3f} kl {row['kl_k3']:.3f} "
                f"pi {row['policy_loss']:.4f} v {row['value_loss']:.4f} "
                f"H {row['entropy']:.2f} clip {row['clip_frac']:.2f} "
                f"({row['tokens_per_sec']:.0f} tok/s)"
            )
        if cfg.plot_every and it and it % cfg.plot_every == 0:
            plot_csv(cfg.log_csv)
        if it and it % cfg.save_every == 0 or it == cfg.num_iters - 1:
            torch.save(
                {"model": policy.state_dict(), "value_head": value_head.state_dict(),
                 "config": model_cfg},
                cfg.save_path,
            )
        it += 1

    print(f"done; saved {cfg.save_path}")


if __name__ == "__main__":
    train_ppo()
