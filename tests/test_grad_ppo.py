"""
tests/test_grad_ppo.py — Module 4.

One test per piece: kl estimators, reward shaping, GAE, policy loss, value loss,
entropy, advantage normalization, rollout alignment.
"""

import torch

from grad_check import check_grad, rel_error
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


# -------------------------------------------------------------------------------------
# Problem 4.1 — gather_logprobs (sanity check, not a grad check per se)
# -------------------------------------------------------------------------------------

def test_gather_logprobs_matches_manual():
    torch.manual_seed(0)
    B, T, V = 2, 4, 7
    logits = torch.randn(B, T, V, dtype=torch.float64)
    targets = torch.randint(0, V, (B, T))
    got = gather_logprobs(logits, targets)
    expect = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(got, expect, atol=1e-12)


# -------------------------------------------------------------------------------------
# Problem 4.2 — KL estimators
# -------------------------------------------------------------------------------------

def test_kl_k1_signed():
    a = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float64)
    b = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    got = kl_k1(a, b)
    assert torch.allclose(got, a, atol=1e-12)


def test_kl_k3_nonnegative():
    torch.manual_seed(0)
    lp = torch.randn(10, dtype=torch.float64)
    ref = torch.randn(10, dtype=torch.float64)
    got = kl_k3(lp, ref)
    assert (got >= -1e-12).all()


# -------------------------------------------------------------------------------------
# Problem 4.3 — reward shaping
# -------------------------------------------------------------------------------------

def test_shape_reward_terminal_only_when_beta_zero():
    B, T = 2, 4
    rm = torch.tensor([1.0, -2.0])
    kl = torch.randn(B, T)
    mask = torch.tensor([
        [1, 1, 1, 1],
        [1, 1, 0, 0],
    ], dtype=torch.float32)
    r = shape_reward(rm, kl, mask, kl_coef=0.0)
    assert r.shape == (B, T)
    # row 0 terminal is position 3; row 1 terminal is position 1
    assert abs(r[0, 3].item() - 1.0) < 1e-12
    assert abs(r[1, 1].item() - (-2.0)) < 1e-12
    # everywhere else (before terminal, with beta=0) is zero
    r[0, 3] = 0.0
    r[1, 1] = 0.0
    assert r.abs().sum().item() < 1e-12


def test_shape_reward_kl_penalty_applied():
    B, T = 1, 3
    rm = torch.tensor([0.0])
    kl = torch.tensor([[0.5, 0.5, 0.5]])
    mask = torch.ones(B, T)
    r = shape_reward(rm, kl, mask, kl_coef=2.0)
    # r_t = -2 * 0.5 = -1.0 at t=0,1; t=2 is terminal: -1 + 0 = -1
    assert torch.allclose(r, torch.full((B, T), -1.0))


# -------------------------------------------------------------------------------------
# Problem 4.4 — GAE
# -------------------------------------------------------------------------------------

def test_gae_three_step_unit():
    rewards = torch.tensor([[1.0, 0.0, 0.0]])
    values = torch.zeros(1, 3)
    mask = torch.ones(1, 3)
    adv, ret = gae(rewards, values, mask, gamma=1.0, lam=1.0)
    assert torch.allclose(adv, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-12)
    assert torch.allclose(ret, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-12)


def test_gae_pad_treated_as_terminal():
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.zeros(1, 3)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    adv, _ = gae(rewards, values, mask, gamma=1.0, lam=1.0)
    # position 2 is pad; from pos 1 the next step is terminal so A_1 = r_1 = 2.0
    # from pos 0: A_0 = r_0 + gamma*lam*A_1 = 1 + 2 = 3
    assert abs(adv[0, 0].item() - 3.0) < 1e-12
    assert abs(adv[0, 1].item() - 2.0) < 1e-12


# -------------------------------------------------------------------------------------
# Problem 4.5 — PPO policy loss
# -------------------------------------------------------------------------------------

def _policy_inputs(seed=0, B=2, T=5):
    torch.manual_seed(seed)
    logp_new = torch.randn(B, T, dtype=torch.float64, requires_grad=True)
    logp_old = logp_new.detach().clone() + 0.01 * torch.randn(B, T, dtype=torch.float64)
    adv = torch.randn(B, T, dtype=torch.float64)
    mask = torch.ones(B, T, dtype=torch.float64)
    return logp_new, logp_old, adv, mask


def test_ppo_policy_loss_grad():
    logp_new, logp_old, adv, mask = _policy_inputs()
    def loss_fn(x):
        return ppo_policy_loss(x, logp_old, adv, mask, clip_eps=0.2)
    rel_err = check_grad(loss_fn, logp_new, atol=1e-5)
    assert rel_err < 1e-5


def test_ppo_policy_loss_clipped_tokens_zero_grad():
    """
    When every ratio is >> 1 + eps AND advantages are positive, the clip is active
    everywhere and the gradient through logp_new must be exactly zero.
    """
    B, T = 2, 4
    logp_old = torch.zeros(B, T, dtype=torch.float64)
    logp_new = torch.full((B, T), 2.0, dtype=torch.float64, requires_grad=True)  # r = e^2 ~ 7.4
    adv = torch.ones(B, T, dtype=torch.float64)  # positive
    mask = torch.ones(B, T, dtype=torch.float64)
    loss = ppo_policy_loss(logp_new, logp_old, adv, mask, clip_eps=0.2)
    loss.backward()
    assert torch.allclose(logp_new.grad, torch.zeros_like(logp_new.grad), atol=1e-12)


# -------------------------------------------------------------------------------------
# Problem 4.6 — value loss
# -------------------------------------------------------------------------------------

def test_value_loss_grad():
    torch.manual_seed(0)
    B, T = 2, 5
    v_new = torch.randn(B, T, dtype=torch.float64, requires_grad=True)
    v_old = v_new.detach() + 0.05 * torch.randn(B, T, dtype=torch.float64)
    returns = torch.randn(B, T, dtype=torch.float64)
    mask = torch.ones(B, T, dtype=torch.float64)
    def loss_fn(x):
        return value_loss(x, v_old, returns, mask, clip_eps_v=0.2)
    rel_err = check_grad(loss_fn, v_new, atol=1e-5)
    assert rel_err < 1e-5


# -------------------------------------------------------------------------------------
# Problem 4.7 — entropy
# -------------------------------------------------------------------------------------

def test_entropy_grad():
    torch.manual_seed(0)
    B, T, V = 2, 4, 7
    logits = torch.randn(B, T, V, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(B, T, dtype=torch.float64)
    def loss_fn(x):
        return masked_entropy(x, mask)
    rel_err = check_grad(loss_fn, logits, atol=1e-5)
    assert rel_err < 1e-5


def test_entropy_ignores_masked():
    torch.manual_seed(0)
    B, T, V = 2, 4, 7
    logits = torch.randn(B, T, V, dtype=torch.float64)
    mask = torch.tensor([
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    ], dtype=torch.float64)
    # perturb logits at masked positions; entropy must not change
    h_a = masked_entropy(logits, mask).item()
    logits_b = logits.clone()
    logits_b[0, 2] += 5.0
    logits_b[1, 3] -= 5.0
    h_b = masked_entropy(logits_b, mask).item()
    assert abs(h_a - h_b) < 1e-12


# -------------------------------------------------------------------------------------
# Problem 4.8 — advantage normalization
# -------------------------------------------------------------------------------------

def test_adv_norm_ignores_pad():
    torch.manual_seed(0)
    B, T = 4, 6
    adv = torch.randn(B, T, dtype=torch.float64)
    mask = torch.ones(B, T, dtype=torch.float64)
    mask[:, -2:] = 0.0
    adv_polluted = adv.clone()
    adv_polluted[:, -2:] = 1e6  # garbage in pad positions
    out_clean = normalize_advantages(adv.clone(), mask)
    out_polluted = normalize_advantages(adv_polluted, mask)
    # both should agree on the real (masked=1) positions
    diff = (out_clean - out_polluted)[mask.bool()].abs().max().item()
    assert diff < 1e-10, f"pad garbage leaked into stats, diff={diff}"

# -------------------------------------------------------------------------------------
# Rollout alignment (provided code, but THE place where off-by-ones hide)
# -------------------------------------------------------------------------------------

def _tiny_policy():
    from config import GPTConfig
    from model import GPT, ScalarHead

    cfg = GPTConfig(block_size=32, vocab_size=32, n_layer=2, n_head=2, n_embd=16)
    return GPT(cfg).eval(), ScalarHead(cfg.n_embd).eval()


def test_rollout_alignment():
    """
    logprobs_old recorded during the rollout must equal an independent recomputation:
    forward over [prompt + response], slice logits[:, T_p-1:-1], gather response ids.
    """
    torch.manual_seed(0)
    policy, value_head = _tiny_policy()
    B, T_p, T_r = 3, 5, 6
    prompt_ids = torch.randint(0, 32, (B, T_p))
    prompt_mask = torch.ones(B, T_p)
    prompt_mask[0, :2] = 0  # one left-padded row

    full_ids, response_ids, logprobs_old, values_old, response_mask = (
        generate_with_logprobs(policy, value_head, prompt_ids, prompt_mask, T_r)
    )
    assert full_ids.shape == (B, T_p + T_r)
    assert response_ids.shape == logprobs_old.shape == values_old.shape == (B, T_r)
    assert torch.all(response_mask == 1.0), "no EOS id given: all tokens are real"

    full_mask = torch.cat([prompt_mask, response_mask], dim=1)
    with torch.no_grad():
        logits = policy(full_ids, attention_mask=full_mask)
    recomputed = gather_logprobs(logits[:, T_p - 1:-1, :], response_ids)
    torch.testing.assert_close(logprobs_old, recomputed, atol=1e-5, rtol=1e-5)


def test_rollout_eos_masking():
    """After a row's first EOS, ids must be EOS padding and the mask must be zero."""
    torch.manual_seed(1)
    policy, value_head = _tiny_policy()
    B, T_p, T_r = 4, 4, 12
    prompt_ids = torch.randint(0, 32, (B, T_p))
    prompt_mask = torch.ones(B, T_p)
    eos = 7

    _, response_ids, _, _, response_mask = generate_with_logprobs(
        policy, value_head, prompt_ids, prompt_mask, T_r, eos_token_id=eos
    )
    for b in range(B):
        ids = response_ids[b].tolist()
        mask = response_mask[b].tolist()
        if eos in ids:
            t = ids.index(eos)
            assert mask[t] == 1.0, "the EOS token itself is a real action"
            assert all(m == 0.0 for m in mask[t + 1:]), "everything after EOS is pad"
            assert all(i == eos for i in ids[t + 1:]), "pad token is EOS"
        else:
            assert all(m == 1.0 for m in mask)
