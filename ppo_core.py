"""
ppo_core.py — PPO building blocks.

Module 4 lives here, one function per problem. Each function gets a gradient check in
tests/test_grad_ppo.py before you integrate it into train_ppo.py.

High-level PPO flow (Module 5 stitches it together):

    ROLLOUT (no grad on policy):
        for each prompt batch:
            response_ids, logprobs_old, values_old, response_mask
                                = generate_with_logprobs(policy, prompts)
            ref_logprobs        = reference_logprobs(ref_model, prompts+responses)
            rm_reward           = reward_model(prompts+responses)
            kl_t                = logprobs_old - ref_logprobs
            per_token_reward    = shape_reward(rm_reward, kl_t, response_mask, kl_coef)
            advantages, returns = gae(per_token_reward, values_old, response_mask,
                                      gamma, lam)

    OPTIMIZE (grad on policy + value):
        for epoch in range(K):
            for mb in minibatches(rollout):
                # Recompute CURRENT logprobs and values on the SAME responses
                logits_new, values_new = policy_and_value(prompts + responses)
                logprobs_new = gather_logprobs(logits_new, responses)
                L_pi = ppo_policy_loss(logprobs_new, logprobs_old, advantages, mask, eps)
                L_v  = value_loss(values_new, values_old, returns, mask, eps_v)
                H    = masked_entropy(logits_new, mask)
                loss = L_pi + c_v * L_v - c_ent * H
                loss.backward(); clip; step
"""

from typing import Tuple

import torch
import torch.nn.functional as F


# =====================================================================================
# Problem 4.1 — Rollout with per-token log-probs
# =====================================================================================

@torch.no_grad()
def generate_with_logprobs(
    policy,
    value_head,
    prompt_ids: torch.Tensor,      # (B, T_p)  LEFT-padded
    prompt_mask: torch.Tensor,     # (B, T_p)  1 on real prompt tokens, 0 on left-pad
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
):
    """
    Generate `max_new_tokens` tokens from `policy`. Record, for EACH generated token,
    the per-token log-prob under the policy at generation time AND the value estimate.

    Returns:
        full_ids       : (B, T_p + T_r)  prompt + generated response (right-padded with
                                          eos_token_id after a row's EOS)
        response_ids   : (B, T_r)        just the generated portion (same padding)
        logprobs_old   : (B, T_r)        log π(a_t | s_t) at generation time
        values_old     : (B, T_r)        V(s_t) at generation time
        response_mask  : (B, T_r)        1 on real generated tokens, 0 after EOS / pad

    Important alignment:
        - At response position t (0-indexed), the policy sees the full prefix
          [prompt, response_{<t}] and produces logits over vocab. We sample a_t, record
          logprob = log softmax(logits)[a_t], and V(s_t) from the value head applied to
          the same hidden state used to predict a_t (i.e. logits-producing hidden at
          the final position of the current input).
        - response_mask[b, t] = 0 for t >= first EOS in row b. Subsequent tokens DO NOT
          contribute to loss (the episode is "done").

    TODO(4.1): implement with a naive per-step forward loop. Clarity > speed here.

    Test (tests/test_grad_ppo.py::test_rollout_alignment):
        fix the seed, compare logprobs_old against a recomputation via
        gather_logprobs(policy(full_ids), shift) — they must match to fp32 tolerance.
    """
    raise NotImplementedError("TODO(4.1): generate_with_logprobs")


def gather_logprobs(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """
    Given logits (B, T, V) and target_ids (B, T), return log p(target_t | prefix) at
    each t using log_softmax + gather. Same shape as target_ids.

    NOTE: this gathers token t from position t's logits — it does NOT shift. If your
    logits come from a forward over [prompt + response] and you want the logprobs of the
    response tokens, slice logits[:, T_p-1:-1, :] and targets = response_ids.

    TODO(4.1): implement (trivial).
    """
    raise NotImplementedError("TODO(4.1): gather_logprobs")


# =====================================================================================
# Problem 4.2 — Per-token KL estimators
# =====================================================================================

def kl_k1(logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
    """
    The InstructGPT / Schulman "k1" KL estimator, per-token:

        kl_t = logprobs_t - ref_logprobs_t       (signed; can be negative on a sample)

    This is an UNBIASED single-sample estimate of KL(π || π_ref) but has nonzero
    variance and can go negative. It's what InstructGPT uses as the shaping penalty.

    TODO(4.2): implement.
    """
    raise NotImplementedError("TODO(4.2): kl_k1")


def kl_k3(logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
    """
    Schulman's "k3" estimator:

        logratio = logprobs - ref_logprobs
        inv_r    = exp(-logratio)
        kl3      = (inv_r - 1) + logratio                  (always >= 0, lower variance)

    Use this for LOGGING (it's strictly nonnegative and visually cleaner). Use k1 as the
    penalty that shapes rewards (unbiased).

    TODO(4.2): implement. Write the derivation in notes/04-ppo-kl.md — why k3 >= 0
    always, and why its gradient is not what you want for the penalty.
    """
    raise NotImplementedError("TODO(4.2): kl_k3")


# =====================================================================================
# Problem 4.3 — Reward shaping
# =====================================================================================

def shape_reward(
    rm_reward: torch.Tensor,       # (B,)  scalar reward at end of response
    kl_t: torch.Tensor,            # (B, T_r) per-token k1 KL
    response_mask: torch.Tensor,   # (B, T_r) 1 on real response tokens
    kl_coef: float,
) -> torch.Tensor:
    """
    Returns per_token_reward: (B, T_r) with

        per_token_reward[b, t] = -kl_coef * kl_t[b, t]           for t < last_response_t
        per_token_reward[b, last_t] += rm_reward[b]              on the final real token

    TODO(4.3): implement.

    Test: with kl_coef = 0 and rm_reward = [1.0, 2.0], per_token_reward should be zero
    everywhere except each row's last real position.
    """
    raise NotImplementedError("TODO(4.3): shape_reward")


# =====================================================================================
# Problem 4.4 — Generalized Advantage Estimation
# =====================================================================================

def gae(
    rewards: torch.Tensor,         # (B, T) per-token rewards
    values: torch.Tensor,          # (B, T) V_t estimates from rollout
    mask: torch.Tensor,            # (B, T) 1 on real tokens, 0 on pad
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE advantages and returns.

    For each row, define V_{T} = 0 (terminal). Then for t = T-1 down to 0:
        δ_t   = r_t + γ · V_{t+1} · nonterm_{t+1} - V_t
        A_t   = δ_t + γ · λ · A_{t+1} · nonterm_{t+1}
    where nonterm_{t+1} = mask[t+1] (treat pad as terminal).

    Returns:
        advantages : (B, T)
        returns    : (B, T) = advantages + values        (value target, stop-grad)

    Important: `returns` is used as the regression target for the value loss. It's a
    function of the CURRENT value estimates, but we treat it as a constant during the
    backward through L_V (this is standard PPO). Do this implicitly by detaching the
    input `values` before the GAE math, or by calling gae inside `torch.no_grad()` —
    the learner should do one of these in train_ppo.py (NOT inside this function;
    this function must remain pure).

    TODO(4.4): implement.

    Test (tests/test_grad_ppo.py::test_gae_three_step):
        T=3, values=[0,0,0], rewards=[1,0,0], mask=all ones, gamma=1, lam=1
          -> advantages = [1, 0, 0], returns = [1, 0, 0].
        Also test with a pad in the middle.
    """
    raise NotImplementedError("TODO(4.4): gae")


# =====================================================================================
# Problem 4.5 — PPO clipped policy loss
# =====================================================================================

def ppo_policy_loss(
    logprobs_new: torch.Tensor,    # (B, T) from current policy (requires_grad=True)
    logprobs_old: torch.Tensor,    # (B, T) from rollout snapshot (no grad)
    advantages: torch.Tensor,      # (B, T) (no grad; normalized per-batch upstream)
    mask: torch.Tensor,            # (B, T) 1 on real response tokens
    clip_eps: float,
) -> torch.Tensor:
    """
    Clipped surrogate objective:

        r_t         = exp(logprobs_new_t - logprobs_old_t)
        surr1_t     = r_t * A_t
        surr2_t     = clip(r_t, 1-eps, 1+eps) * A_t
        per_token_L = -min(surr1_t, surr2_t)
        return (per_token_L * mask).sum() / mask.sum().clamp_min(1.0)

    Gradient intuition (derive in notes/04-ppo-policy.md):
        - Where the clip is INACTIVE (r in [1-eps, 1+eps], or A says we'd clip the
          wrong way), gradient of the policy wrt logprobs_new_t is -A_t * r_t.
        - Where the clip IS active AND it lowers the objective, gradient is zero
          (clipping throws the token's contribution away).
        - Sign flips with sign(A_t). A_t > 0 (good action) pushes logprob UP; A_t < 0
          pushes it down. Classic policy gradient.

    TODO(4.5): implement.

    Tests:
        1) random fp64 inputs, grad check vs autograd on logprobs_new.
        2) set all ratios to 1.5 with eps=0.2 and A>0 → each token contributes
           `-1.2 * A`; check numerically.
        3) set all ratios WAY outside [1-eps, 1+eps] in the "penalizing" direction
           (positive A and r >> 1) → the min branch is the clipped one → gradient on
           logprobs_new must be exactly zero on those tokens.
    """
    raise NotImplementedError("TODO(4.5): ppo_policy_loss")


# =====================================================================================
# Problem 4.6 — Clipped value loss
# =====================================================================================

def value_loss(
    values_new: torch.Tensor,      # (B, T) current value estimates (requires_grad)
    values_old: torch.Tensor,      # (B, T) rollout-time values (no grad)
    returns: torch.Tensor,         # (B, T) GAE returns (no grad)
    mask: torch.Tensor,            # (B, T)
    clip_eps_v: float,
) -> torch.Tensor:
    """
        v_clipped = values_old + clip(values_new - values_old, -eps_v, eps_v)
        per_t     = 0.5 * max( (values_new - returns)^2, (v_clipped - returns)^2 )
        return (per_t * mask).sum() / mask.sum().clamp_min(1.0)

    Why clip: the value net trains faster than the policy during PPO and can overshoot
    early, which wrecks the advantage estimates used by the policy loss. Clipping
    prevents giant value jumps inside one rollout batch.

    TODO(4.6): implement + grad check.
    """
    raise NotImplementedError("TODO(4.6): value_loss")


# =====================================================================================
# Problem 4.7 — Masked entropy bonus
# =====================================================================================

def masked_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean per-token categorical entropy over MASKED (real) positions only.

        p     = softmax(logits, dim=-1)
        H_t   = -(p * log p).sum(-1)               # (B, T)
        return (H_t * mask).sum() / mask.sum().clamp_min(1.0)

    Numerical stability: use `logits - logits.max(-1, keepdim=True).values` before the
    softmax, OR use `F.log_softmax` and compute H = -(p * logp).sum(-1).

    TODO(4.7): implement + grad check.
    """
    raise NotImplementedError("TODO(4.7): masked_entropy")


# =====================================================================================
# Problem 4.8 — Advantage normalization
# =====================================================================================

def normalize_advantages(advantages: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Center and scale advantages using statistics computed over REAL (masked) tokens only.

        n     = mask.sum().clamp_min(1.0)
        mean  = (advantages * mask).sum() / n
        var   = ((advantages - mean) ** 2 * mask).sum() / n
        return (advantages - mean) / (var.sqrt() + eps)

    Careful: do NOT let pad-token garbage pollute the mean/std.

    TODO(4.8): implement.

    Test (tests/test_grad_ppo.py::test_adv_norm_ignores_pad):
        Create advantages where the PAD positions are 1000.0 but the mask zeros them.
        Verify the normalized stats equal those you'd get if you instead manually
        dropped the pad positions before computing mean/std.
    """
    raise NotImplementedError("TODO(4.8): normalize_advantages")
