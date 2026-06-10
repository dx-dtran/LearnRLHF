"""
ppo_core.py — the PPO building blocks. This is the core of the course.

Part 4 of the crash course. Your jobs (one function per blank, each with its own
test in tests/test_grad_ppo.py):

    [FILL 4.1]  gather_logprobs        — per-token log p(target) from logits
    [FILL 4.2]  kl_k1, kl_k3           — per-token KL estimators
    [FILL 4.3]  shape_reward           — terminal RM reward + per-token KL penalty
    [FILL 4.4]  gae                    — generalized advantage estimation
    [FILL 4.5]  ppo_policy_loss        — clipped surrogate
    [FILL 4.6]  value_loss             — clipped value regression
    [FILL 4.7]  masked_entropy         — entropy bonus over real tokens
    [FILL 4.8]  normalize_advantages   — masked mean/std normalization

`generate_with_logprobs` (the rollout) is provided: it is fiddly alignment code, and
the test test_rollout_alignment pins it down. Read it carefully — the off-by-one it
handles (logits at position t score token t+1) is exactly the one you must respect
in gather_logprobs.

How the pieces flow together (train_ppo.py is the glue):

    ROLLOUT, no grad:
        full_ids, response_ids, logprobs_old, values_old, response_mask
                          = generate_with_logprobs(policy, value_head, prompts, ...)
        ref_logprobs      = gather_logprobs(ref(full_ids)[:, T_p-1:-1], response_ids)
        rm_reward         = reward_model(full_ids, mask, last_idx)         # (B,)
        kl                = kl_k1(logprobs_old, ref_logprobs)
        rewards           = shape_reward(rm_reward, kl, response_mask, beta)
        advantages, rets  = gae(rewards, values_old, response_mask, gamma, lam)
        advantages        = normalize_advantages(advantages, response_mask)

    OPTIMIZE, K epochs of minibatches, grad on policy + value head:
        recompute logprobs_new, values_new on the SAME responses
        L = ppo_policy_loss(...) + c_v * value_loss(...) - c_ent * masked_entropy(...)
"""

from typing import Tuple

import torch
import torch.nn.functional as F


# =====================================================================================
# [FILL 4.1] — per-token log-probs
# =====================================================================================


def gather_logprobs(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """
    logits (B, T, V), target_ids (B, T) -> log p(target_t) at each position, (B, T).

    No shifting here: this scores target_ids[t] under logits[t]. The CALLER chooses
    the slice. To score response tokens from a forward over [prompt + response],
    pass logits[:, T_p-1:-1, :] (logits at position i predict token i+1).
    """
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


# =====================================================================================
# Rollout (provided)
# =====================================================================================


@torch.no_grad()
def generate_with_logprobs(
    policy,
    value_head,
    prompt_ids: torch.Tensor,    # (B, T_p) LEFT-padded
    prompt_mask: torch.Tensor,   # (B, T_p) 1 on real prompt tokens
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
):
    """
    Sample responses and record, per generated token, the log-prob under the policy
    and the value estimate of the state it was sampled from.

    Returns:
        full_ids      (B, T_p + T_r)  prompt + response, response right-padded w/ EOS
        response_ids  (B, T_r)
        logprobs_old  (B, T_r)  log pi(a_t | s_t)
        values_old    (B, T_r)  V(s_t), the state BEFORE emitting token t
        response_mask (B, T_r)  1 on real tokens up to and including the first EOS

    Two-phase design, simpler than recording during the loop:
      1. Sample autoregressively (one forward per step, no KV cache).
      2. ONE forward over [prompt + response] recomputes everything. Deterministic
         forward + dropout 0 means the recomputed numbers equal the rollout-time
         numbers, and the slice logic lives in exactly one place:
             logits[:, T_p-1+t] scored response token t when it was sampled,
             so logprobs/values for the response live at columns T_p-1 .. T-2.

    Keep temperature=1.0 / no top-k / no top-p for PPO: logprobs_old must describe
    the distribution tokens were ACTUALLY drawn from, and we record the raw softmax.
    """
    from model import filter_logits

    B, T_p = prompt_ids.shape
    device = prompt_ids.device

    ids = prompt_ids
    mask = prompt_mask.clone().float()
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        logits = policy(ids, attention_mask=mask)[:, -1, :] / max(temperature, 1e-8)
        logits = filter_logits(logits, top_k=top_k, top_p=top_p)
        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
        if eos_token_id is not None:
            next_id = torch.where(
                finished, torch.full_like(next_id, eos_token_id), next_id
            )
        # the token is real iff the row was still running when it was sampled;
        # an EOS sampled by a running row is real (it is the "stop" action)
        new_bit = (~finished).float()
        ids = torch.cat([ids, next_id.unsqueeze(1)], dim=1)
        mask = torch.cat([mask, new_bit.unsqueeze(1)], dim=1)
        if eos_token_id is not None:
            finished = finished | (next_id == eos_token_id)

    response_ids = ids[:, T_p:]
    response_mask = mask[:, T_p:]

    # one clean recompute for logprobs + values
    hidden = policy.forward_hidden(ids, attention_mask=mask)
    logits = hidden @ policy.wte.weight.t()
    logprobs_old = gather_logprobs(logits[:, T_p - 1:-1, :], response_ids)
    values_old = value_head(hidden)[:, T_p - 1:-1]

    return ids, response_ids, logprobs_old, values_old, response_mask


# =====================================================================================
# [FILL 4.2] — per-token KL estimators
# =====================================================================================


def kl_k1(logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
    """
    The "k1" estimator, used by InstructGPT as the reward-shaping penalty:

        kl_t = logprobs_t - ref_logprobs_t

    Unbiased single-sample estimate of KL(pi || pi_ref); signed, can be negative on
    any one token.
    """
    return logprobs - ref_logprobs


def kl_k3(logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
    """
    Schulman's "k3" estimator, for LOGGING (always >= 0, lower variance):

        logratio = logprobs - ref_logprobs
        kl3      = exp(-logratio) - 1 + logratio

    Note the direction: with samples from pi, the estimator uses the INVERSE ratio
    pi_ref/pi = exp(-logratio). It is >= 0 because e^x - 1 - x >= 0 for all x.
    """
    logratio = logprobs - ref_logprobs
    return torch.exp(-logratio) - 1 + logratio


# =====================================================================================
# [FILL 4.3] — reward shaping
# =====================================================================================


def shape_reward(
    rm_reward: torch.Tensor,      # (B,) scalar reward for the whole response
    kl_t: torch.Tensor,           # (B, T_r) per-token k1 KL
    response_mask: torch.Tensor,  # (B, T_r) 1 on real response tokens
    kl_coef: float,
) -> torch.Tensor:
    """
    Per-token reward (B, T_r):

        r_t = -kl_coef * kl_t                 on every real token
        r_t += rm_reward                      on each row's LAST real token only

    Last real index = response_mask.sum(-1) - 1 (responses always have >= 1 real
    token). Pad positions get reward 0.
    """
    rewards = -kl_coef * kl_t * response_mask
    last_idx = (response_mask.sum(dim=1).long() - 1).clamp_min(0)
    rewards = rewards.clone()
    rewards[torch.arange(rewards.size(0), device=rewards.device), last_idx] += rm_reward
    return rewards


# =====================================================================================
# [FILL 4.4] — Generalized Advantage Estimation
# =====================================================================================


def gae(
    rewards: torch.Tensor,  # (B, T) per-token rewards
    values: torch.Tensor,   # (B, T) V_t from the rollout
    mask: torch.Tensor,     # (B, T) 1 on real tokens
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward recursion, treating pad as terminal (V after the last real token = 0):

        nonterm_t+1 = mask[t+1]            (and 0 at t = T-1)
        delta_t = r_t + gamma * V_{t+1} * nonterm_{t+1} - V_t
        A_t     = delta_t + gamma * lam * A_{t+1} * nonterm_{t+1}

    Returns (advantages, returns) with returns = advantages + values: that is the
    regression target for the value head, and it is treated as a CONSTANT there
    (train_ppo computes all of this under no_grad).

    A plain python loop over t from T-1 down to 0 is exactly right here.
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    next_adv = torch.zeros(B, dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(T)):
        nonterm = mask[:, t + 1] if t + 1 < T else torch.zeros_like(mask[:, t])
        next_value = values[:, t + 1] if t + 1 < T else torch.zeros_like(values[:, t])
        delta = rewards[:, t] + gamma * next_value * nonterm - values[:, t]
        next_adv = delta + gamma * lam * next_adv * nonterm
        advantages[:, t] = next_adv
    returns = advantages + values
    return advantages, returns


# =====================================================================================
# [FILL 4.5] — PPO clipped policy loss
# =====================================================================================


def ppo_policy_loss(
    logprobs_new: torch.Tensor,  # (B, T) current policy (requires grad)
    logprobs_old: torch.Tensor,  # (B, T) rollout snapshot (no grad)
    advantages: torch.Tensor,    # (B, T) no grad, already normalized
    mask: torch.Tensor,          # (B, T) 1 on real response tokens
    clip_eps: float,
) -> torch.Tensor:
    """
    The clipped surrogate, averaged over real tokens:

        ratio_t = exp(logprobs_new_t - logprobs_old_t)
        surr1_t = ratio_t * A_t
        surr2_t = clamp(ratio_t, 1 - eps, 1 + eps) * A_t
        L       = -(min(surr1, surr2) * mask).sum() / mask.sum().clamp_min(1.0)

    Gradient (derive before coding — notes/04-ppo-policy.md):
        unclipped tokens:  dL/dlogprobs_new_t = -A_t * ratio_t / N
        clipped tokens (min picks the clamped branch): exactly zero.
    """
    ratio = torch.exp(logprobs_new - logprobs_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    per_token = -torch.min(surr1, surr2)
    return (per_token * mask).sum() / mask.sum().clamp_min(1.0)


# =====================================================================================
# [FILL 4.6] — clipped value loss
# =====================================================================================


def value_loss(
    values_new: torch.Tensor,  # (B, T) current value head (requires grad)
    values_old: torch.Tensor,  # (B, T) rollout values (no grad)
    returns: torch.Tensor,     # (B, T) GAE returns (no grad)
    mask: torch.Tensor,        # (B, T)
    clip_eps_v: float,
) -> torch.Tensor:
    """
        v_clip = values_old + clamp(values_new - values_old, -eps_v, eps_v)
        per_t  = 0.5 * max((values_new - returns)^2, (v_clip - returns)^2)
        L      = (per_t * mask).sum() / mask.sum().clamp_min(1.0)

    The clip keeps the value head from jumping far from its rollout-time estimates
    within one PPO phase; early in training the value loss otherwise dominates and
    destabilizes the advantages computed from it.
    """
    v_clip = values_old + torch.clamp(values_new - values_old, -clip_eps_v, clip_eps_v)
    per_token = 0.5 * torch.maximum(
        (values_new - returns) ** 2, (v_clip - returns) ** 2
    )
    return (per_token * mask).sum() / mask.sum().clamp_min(1.0)


# =====================================================================================
# [FILL 4.7] — masked entropy bonus
# =====================================================================================


def masked_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean per-token entropy over real positions:

        logp = log_softmax(logits, dim=-1)
        H_t  = -(exp(logp) * logp).sum(-1)                    # (B, T)
        return (H_t * mask).sum() / mask.sum().clamp_min(1.0)

    log_softmax keeps it numerically stable. Gradient, for the derivation:
    dH/dlogits = -p * (logp + H), elementwise per position.
    """
    logp = F.log_softmax(logits, dim=-1)
    h = -(logp.exp() * logp).sum(dim=-1)
    return (h * mask).sum() / mask.sum().clamp_min(1.0)


# =====================================================================================
# [FILL 4.8] — advantage normalization over real tokens only
# =====================================================================================


def normalize_advantages(
    advantages: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Subtract the mean and divide by the std, both computed over REAL tokens only:

        n    = mask.sum().clamp_min(1.0)
        mean = (advantages * mask).sum() / n
        var  = ((advantages - mean)^2 * mask).sum() / n
        out  = (advantages - mean) / (sqrt(var) + eps)

    Getting this mask wrong is the single most common PPO bug: one batch of garbage
    pad values in the std and every advantage in the batch is silently rescaled.
    """
    n = mask.sum().clamp_min(1.0)
    mean = (advantages * mask).sum() / n
    var = (((advantages - mean) ** 2) * mask).sum() / n
    return (advantages - mean) / (var.sqrt() + eps)
