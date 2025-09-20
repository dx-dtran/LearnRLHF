import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from gpt import GPTConfig, GPT
from train_ppo import ScalarHead, PPOTrainer, gather_log_probs, ppo_policy_objective


def build_models():
    config = GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    policy = GPT(config)
    reference = GPT(config)
    value = ScalarHead(config)
    reward = ScalarHead(config)
    reference.load_state_dict(policy.state_dict())
    value.body.load_state_dict(policy.state_dict())
    reward.body.load_state_dict(policy.state_dict())
    return config, policy, reference, value, reward


def test_ppo_step_runs():
    config, policy, reference, value, reward = build_models()
    trainer = PPOTrainer(policy, reference, value, reward, pad_token=0, lr=1e-4)

    batch = 2
    prompt_len = 4
    response_len = 3
    full = torch.randint(0, config.vocab_size, (batch, prompt_len + response_len))
    mask = torch.ones_like(full)
    response_mask = torch.zeros_like(full)
    response_mask[:, prompt_len:] = 1
    responses = full[:, prompt_len:]
    responses_mask = torch.ones(batch, response_len)
    prompt_lengths = torch.full((batch,), prompt_len)
    response_lengths = torch.full((batch,), response_len)

    with torch.no_grad():
        logits, _ = policy(full, attention_mask=mask)
        log_probs = F.log_softmax(logits, dim=-1)
        old_log_probs = gather_log_probs(
            log_probs, responses, prompt_lengths, response_lengths
        )
        old_values = value(full, mask)

    rewards = torch.randn(batch)
    advantages = torch.randn(batch)

    data = {
        "full": full,
        "full_mask": mask,
        "response_mask": response_mask,
        "responses": responses,
        "responses_mask": responses_mask,
        "prompt_lengths": prompt_lengths,
        "response_lengths": response_lengths,
        "old_log_probs": old_log_probs,
        "old_values": old_values,
        "advantages": advantages,
        "rewards": rewards,
    }

    metrics = trainer.train_step(data)
    assert "policy_loss" in metrics
    assert "value_loss" in metrics


def test_ppo_policy_gradient_matches_finite_difference():
    torch.manual_seed(0)
    batch = 1
    prompt_len = 2
    response_len = 1
    vocab = 5

    log_probs = torch.randn(batch, prompt_len + response_len, vocab, requires_grad=True)
    ref_log_probs = torch.randn_like(log_probs)
    responses = torch.tensor([[3]], dtype=torch.long)
    responses_mask = torch.ones(batch, response_len)
    prompt_lengths = torch.tensor([prompt_len])
    response_lengths = torch.tensor([response_len])
    response_token_mask = torch.zeros(batch, prompt_len + response_len)
    response_token_mask[:, prompt_len:] = 1
    old_log_probs = torch.randn(batch, response_len)
    advantages = torch.tensor([0.5])

    total, _, _, _ = ppo_policy_objective(
        log_probs,
        ref_log_probs,
        responses,
        responses_mask,
        prompt_lengths,
        response_lengths,
        response_token_mask,
        old_log_probs,
        advantages,
        clip=0.2,
        kl_coeff=0.1,
        entropy_coeff=0.01,
    )
    total.backward()

    idx = (0, prompt_len, 3)
    autograd_grad = log_probs.grad[idx].item()

    eps = 1e-3
    base = log_probs.detach().clone()
    plus = base.clone()
    plus[idx] += eps
    minus = base.clone()
    minus[idx] -= eps

    plus_total = ppo_policy_objective(
        plus,
        ref_log_probs,
        responses,
        responses_mask,
        prompt_lengths,
        response_lengths,
        response_token_mask,
        old_log_probs,
        advantages,
        clip=0.2,
        kl_coeff=0.1,
        entropy_coeff=0.01,
    )[0].item()

    minus_total = ppo_policy_objective(
        minus,
        ref_log_probs,
        responses,
        responses_mask,
        prompt_lengths,
        response_lengths,
        response_token_mask,
        old_log_probs,
        advantages,
        clip=0.2,
        kl_coeff=0.1,
        entropy_coeff=0.01,
    )[0].item()

    finite = (plus_total - minus_total) / (2 * eps)
    assert torch.isclose(torch.tensor(autograd_grad), torch.tensor(finite), atol=1e-3, rtol=1e-3)
