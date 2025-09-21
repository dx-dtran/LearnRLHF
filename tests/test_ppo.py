import os
import sys

import torch
import torch.nn.functional as F
from torch.autograd import gradcheck

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from gpt import GPTConfig, GPT
from reward_model import ScalarHead
from train_ppo import PPOTrainer, gather_log_probs
from train_ppo import train_ppo  # noqa: F401  # ensure module imports without argparse


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


def test_ppo_policy_loss_gradcheck():
    torch.manual_seed(0)
    config, policy, reference, value, reward = build_models()
    trainer = PPOTrainer(policy, reference, value, reward, pad_token=0, lr=1e-4)

    batch = 1
    prompt_len = 2
    response_len = 3
    vocab = config.vocab_size

    log_probs = torch.randn(
        batch, prompt_len + response_len, vocab, dtype=torch.double, requires_grad=True
    )
    ref_log_probs = torch.randn(batch, prompt_len + response_len, vocab, dtype=torch.double)
    responses = torch.randint(0, vocab, (batch, response_len), dtype=torch.long)
    prompt_lengths = torch.tensor([prompt_len], dtype=torch.double)
    response_lengths = torch.tensor([response_len], dtype=torch.double)
    responses_mask = torch.ones(batch, response_len, dtype=torch.double)
    full_response_mask = torch.zeros(batch, prompt_len + response_len, dtype=torch.double)
    full_response_mask[:, prompt_len:] = 1.0
    advantages = torch.randn(batch, dtype=torch.double)
    old_log_probs = torch.randn(batch, response_len, dtype=torch.double)

    def objective(log_probs_input: torch.Tensor) -> torch.Tensor:
        policy_loss, kl, entropy = trainer.compute_policy_losses(
            log_probs_input,
            ref_log_probs,
            responses,
            prompt_lengths,
            response_lengths,
            responses_mask,
            full_response_mask,
            advantages,
            old_log_probs,
        )
        return policy_loss + trainer.kl * kl - trainer.entropy * entropy

    assert gradcheck(objective, (log_probs,), eps=1e-6, atol=1e-4, rtol=1e-4)
