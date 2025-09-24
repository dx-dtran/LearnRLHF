from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import PreferenceDataset
from gpt import GPT, GPTConfig
from train_rm import ScalarHead
from ppo_data import (
    batch_prompt_rows,
    build_model_inputs,
    gather_response_log_probs,
)

class ValueHead(nn.Module):
    """Lightweight value projection sharing the policy transformer."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if hidden.size(0) == 0:
            return hidden.new_zeros((0,))
        lengths = mask.sum(dim=1).long().clamp(min=1) - 1
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = hidden[batch_indices, lengths]
        return self.proj(last_hidden).squeeze(-1)

class PPOTrainer:
    def __init__(
        self,
        policy: GPT,
        reference: GPT,
        value_head: ValueHead,
        reward: ScalarHead,
        pad_token: int,
        clip: float = 0.2,
        kl: float = 0.1,
        entropy: float = 0.01,
        lr: float = 1e-5,
    ) -> None:
        self.policy = policy
        self.reference = reference
        self.value_head = value_head
        self.reward = reward
        self.pad = pad_token
        self.clip = clip
        self.kl = kl
        self.entropy = entropy

        for param in self.reference.parameters():
            param.requires_grad_(False)
        for param in self.reward.parameters():
            param.requires_grad_(False)

        self.policy_opt = torch.optim.AdamW(self.policy.parameters(), lr=lr)

    def compute_values(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.value_head(hidden, mask)

    def compute_policy_losses(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        responses: torch.Tensor,
        prompt_lengths: torch.Tensor,
        response_lengths: torch.Tensor,
        responses_mask: torch.Tensor,
        full_response_mask: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_log_probs = gather_response_log_probs(
            log_probs, responses, prompt_lengths, response_lengths
        )
        ratios = (new_log_probs - old_log_probs).exp()
        advantages = advantages.unsqueeze(1)

        unclipped = ratios * advantages
        clipped = torch.clamp(ratios, 1.0 - self.clip, 1.0 + self.clip) * advantages
        mask = responses_mask
        denom = mask.sum().clamp(min=1.0)
        policy_loss = -torch.sum(torch.minimum(unclipped, clipped) * mask) / denom

        probs = log_probs.exp()
        kl_per_token = (probs * (log_probs - ref_log_probs)).sum(dim=-1)
        kl = (kl_per_token * full_response_mask).sum() / full_response_mask.sum().clamp(min=1.0)

        entropy_per_token = -(probs * log_probs).sum(dim=-1)
        entropy = (entropy_per_token * full_response_mask).sum() / full_response_mask.sum().clamp(min=1.0)

        return policy_loss, kl, entropy

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        full = batch["full"]
        full_mask = batch["full_mask"]
        responses = batch["responses"]
        responses_mask = batch["responses_mask"]
        prompt_lengths = batch["prompt_lengths"]
        response_lengths = batch["response_lengths"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]
        old_values = batch["old_values"]
        rewards = batch["rewards"]
        response_mask = batch["response_mask"]

        hidden = self.policy.transform(full, attention_mask=full_mask)
        logits = self.policy.head(hidden)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            ref_logits, _ = self.reference(full, attention_mask=full_mask)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        policy_loss, kl, entropy = self.compute_policy_losses(
            log_probs,
            ref_log_probs,
            responses,
            prompt_lengths,
            response_lengths,
            responses_mask,
            response_mask,
            advantages,
            old_log_probs,
        )

        policy_objective = policy_loss + self.kl * kl - self.entropy * entropy
        predicted_values = self.compute_values(hidden, full_mask)
        target_values = rewards.to(predicted_values.device)
        value_loss = torch.nn.functional.mse_loss(predicted_values, target_values)

        total_loss = policy_objective + value_loss
        self.policy_opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_opt.step()

        improvement = (
            predicted_values.detach() - old_values.to(predicted_values.device)
        ).mean()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "kl": float(kl.item()),
            "entropy": float(entropy.item()),
            "value_improvement": float(improvement.item()),
            "reward": float(rewards.mean().item()),
        }


def train_ppo(
    preference_path: str,
    *,
    reward_path: str,
    policy_init: Optional[str] = None,
    out_path: str = "weights/ppo_policy.pt",
    batch_size: int = 4,
    epochs: int = 1,
    max_new_tokens: int = 64,
    lr: float = 1e-5,
    clip: float = 0.2,
    kl_coef: float = 0.1,
    entropy_coef: float = 0.01,
    device: Optional[torch.device] = None,
) -> str:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(preference_path):
        raise FileNotFoundError(
            f"Preference data not found at {preference_path}. Prepare the JSONL file before training."
        )
    if not os.path.exists(reward_path):
        raise FileNotFoundError(
            f"Reward model weights not found at {reward_path}. Train the reward model first."
        )

    config = GPTConfig()
    dataset = PreferenceDataset(preference_path, block_size=config.block_size)
    tokenizer_bundle = dataset.tokenizer_bundle
    if tokenizer_bundle.encoder.n_vocab != config.vocab_size:
        if tokenizer_bundle.encoder.n_vocab > config.vocab_size:
            raise ValueError("Tokenizer vocabulary is larger than the model embedding size")
        config.vocab_size = tokenizer_bundle.encoder.n_vocab

    policy = GPT(config)
    reference = GPT(config)
    reward_model = ScalarHead(config)

    if policy_init is not None:
        if not os.path.exists(policy_init):
            raise FileNotFoundError(
                f"Initial policy checkpoint {policy_init} does not exist; provide a Torch state dict"
            )
        policy_state = torch.load(policy_init, map_location="cpu")
        policy.load_state_dict(policy_state, strict=False)

    reference.load_state_dict(policy.state_dict())
    reward_state = torch.load(reward_path, map_location="cpu")
    reward_model.load_state_dict(reward_state, strict=False)

    value_head = ValueHead(config.n_embd)
    policy.add_module("value_head", value_head)

    policy.to(device)
    reference.to(device)
    reward_model.to(device)

    trainer = PPOTrainer(
        policy,
        reference,
        value_head,
        reward_model,
        pad_token=tokenizer_bundle.pad,
        clip=clip,
        kl=kl_coef,
        entropy=entropy_coef,
        lr=lr,
    )

    for epoch in range(epochs):
        order = torch.randperm(len(dataset))
        for start in range(0, len(dataset), batch_size):
            indices = order[start : start + batch_size]
            rows = [dataset[int(i)] for i in indices]
            if not rows:
                continue
            prompt_tokens, prompt_mask = batch_prompt_rows(rows, tokenizer_bundle.pad)
            prompt_tokens = prompt_tokens.to(device)
            prompt_mask = prompt_mask.to(device)

            sample = build_model_inputs(
                policy,
                policy.value_head,
                prompt_tokens,
                prompt_mask,
                max_new_tokens=max_new_tokens,
                pad_token=tokenizer_bundle.pad,
                eos_token=tokenizer_bundle.eos,
            )
            with torch.no_grad():
                rewards = reward_model(sample["full"], sample["full_mask"])
            advantages = rewards - sample["old_values"]
            if advantages.numel():
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

            sample["advantages"] = advantages.detach()
            sample["rewards"] = rewards.detach()

            metrics = trainer.train_step(sample)
            print(
                f"epoch {epoch + 1} step {start // batch_size + 1}: "
                f"reward={metrics['reward']:.3f} policy_loss={metrics['policy_loss']:.3f}"
            )

    if out_path:
        directory = os.path.dirname(out_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(policy.state_dict(), out_path)
    return out_path


def main() -> None:
    preference_path = "data/hh_rlhf_preferences_train.jsonl"
    reward_path = "weights/reward_model.pt"
    policy_init = "weights/sft.pt"
    out_path = "weights/ppo_policy.pt"
    batch_size = 8
    epochs = 4
    max_new_tokens = 64
    lr = 1e-5
    clip = 0.2
    kl_coef = 0.1
    entropy_coef = 0.01

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ppo(
        preference_path,
        reward_path=reward_path,
        policy_init=policy_init,
        out_path=out_path,
        batch_size=batch_size,
        epochs=epochs,
        max_new_tokens=max_new_tokens,
        lr=lr,
        clip=clip,
        kl_coef=kl_coef,
        entropy_coef=entropy_coef,
        device=device,
    )


if __name__ == "__main__":
    main()
