"""
tests/test_training_smoke.py — does the machinery actually TRAIN?

Three end-to-end optimization tests on tiny models with synthetic data, CPU-only,
no downloads. Each one runs the same code paths the real training scripts use and
asserts that the objective measurably improves. If these pass, the remaining risk
in a real run is data and hyperparameters, not math.

    1. SFT:  a tiny GPT learns a deterministic token pattern (loss drops 50%+).
    2. RM:   a tiny reward model learns a synthetic preference rule (acc > 0.9).
    3. PPO:  a tiny policy learns to emit a target token the "reward model" pays
             for (mean reward at the end far above the random-policy start).

These take a few seconds each. They are the closest thing to a proof that a real
GPU run will move the model in the right direction.
"""

import torch

from config import GPTConfig, PPOConfig
from model import GPT, ScalarHead

TINY = GPTConfig(block_size=32, vocab_size=32, n_layer=2, n_head=2, n_embd=32)


def test_sft_smoke_loss_decreases():
    from train_sft import build_optimizer, sft_loss

    torch.manual_seed(0)
    model = GPT(TINY).train()
    optim = build_optimizer(model, lr=3e-3, weight_decay=0.0, betas=(0.9, 0.95))

    # deterministic pattern: next token = (token + 1) % V, supervised on 2nd half
    B, T = 8, 16
    start = torch.randint(0, TINY.vocab_size, (B, 1))
    input_ids = (start + torch.arange(T)) % TINY.vocab_size
    labels = (input_ids + 1) % TINY.vocab_size
    loss_mask = torch.zeros(B, T)
    loss_mask[:, T // 2:] = 1.0

    losses = []
    for _ in range(60):
        logits = model(input_ids)
        loss = sft_loss(logits, labels, loss_mask)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        losses.append(loss.item())

    assert losses[-1] < 0.5 * losses[0], (
        f"SFT smoke run failed to learn: first={losses[0]:.3f} last={losses[-1]:.3f}"
    )


def test_rm_smoke_learns_preference():
    from train_rm import RewardModel, bt_loss, pairwise_accuracy
    from train_sft import build_optimizer

    torch.manual_seed(0)
    model = RewardModel(TINY).train()
    optim = build_optimizer(model, lr=1e-3, weight_decay=0.0, betas=(0.9, 0.95))

    GOOD, BAD = 7, 3

    def make_batch(B=16, T=10):
        chosen = torch.randint(0, TINY.vocab_size, (B, T))
        rejected = torch.randint(0, TINY.vocab_size, (B, T))
        chosen[:, -1] = GOOD
        rejected[:, -1] = BAD
        mask = torch.ones(B, T)
        last_idx = torch.full((B,), T - 1, dtype=torch.long)
        return chosen, rejected, mask, last_idx

    for _ in range(60):
        c, r, mask, last = make_batch()
        loss = bt_loss(model(c, mask, last), model(r, mask, last))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        c, r, mask, last = make_batch(B=64)
        acc = pairwise_accuracy(model(c, mask, last), model(r, mask, last)).item()
    assert acc > 0.9, f"RM smoke run failed to learn the preference rule: acc={acc:.2f}"


class _CountTokenRM:
    """Stand-in reward model: reward = fraction of response tokens equal to TARGET."""

    def __init__(self, target: int, T_p: int):
        self.target = target
        self.T_p = T_p

    def __call__(self, full_ids, attention_mask, last_idx):
        response = full_ids[:, self.T_p:]
        resp_mask = attention_mask[:, self.T_p:]
        hits = ((response == self.target).float() * resp_mask).sum(dim=1)
        return hits / resp_mask.sum(dim=1).clamp_min(1.0)


def test_ppo_smoke_reward_increases():
    from train_ppo import optimize, rollout

    torch.manual_seed(0)
    cfg = PPOConfig(
        response_max_len=8,
        rollout_batch_size=16,
        minibatch_size=8,
        ppo_epochs=4,
        policy_lr=1e-3,
        value_lr=3e-3,
        kl_coef=0.01,
        temperature=1.0,
    )
    policy = GPT(TINY).train()
    value_head = ScalarHead(TINY.n_embd).train()
    ref = GPT(TINY).eval().requires_grad_(False)
    ref.load_state_dict(policy.state_dict())

    B, T_p = cfg.rollout_batch_size, 4
    rm = _CountTokenRM(target=5, T_p=T_p)

    optimizer = torch.optim.AdamW(
        [
            {"params": policy.parameters(), "lr": cfg.policy_lr},
            {"params": value_head.parameters(), "lr": cfg.value_lr},
        ],
        betas=cfg.betas,
    )

    rewards = []
    for it in range(20):
        prompt_ids = torch.randint(0, TINY.vocab_size, (B, T_p))
        prompt_mask = torch.ones(B, T_p)
        ro = rollout(policy, value_head, ref, rm, prompt_ids, prompt_mask, cfg, None)
        stats = ro.pop("stats")
        optimize(policy, value_head, optimizer, ro, cfg)
        rewards.append(stats["rm_reward"])

    first, last = sum(rewards[:3]) / 3, sum(rewards[-3:]) / 3
    assert last > first + 0.1, (
        f"PPO smoke run failed to improve reward: first iters {first:.3f}, "
        f"last iters {last:.3f}, trace={['%.2f' % r for r in rewards]}"
    )
