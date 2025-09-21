from __future__ import annotations

import torch
import torch.nn.functional as F

from gpt import GPT
from train_rm import ScalarHead


def gather_log_probs(
    log_probs: torch.Tensor,
    responses: torch.Tensor,
    prompt_lengths: torch.Tensor,
    response_lengths: torch.Tensor,
) -> torch.Tensor:
    batch, max_response = responses.shape
    collected = log_probs.new_zeros(batch, max_response)
    for i in range(batch):
        start = int(prompt_lengths[i].item())
        length = int(response_lengths[i].item())
        if length == 0:
            continue
        token_slice = responses[i, :length]
        log_slice = log_probs[i, start : start + length]
        collected[i, :length] = log_slice.gather(1, token_slice.unsqueeze(-1)).squeeze(-1)
    return collected


class PPOTrainer:
    def __init__(
        self,
        policy: GPT,
        reference: GPT,
        value: ScalarHead,
        reward: ScalarHead,
        pad_token: int,
        clip: float = 0.2,
        kl: float = 0.1,
        entropy: float = 0.01,
        lr: float = 1e-5,
    ) -> None:
        self.policy = policy
        self.reference = reference
        self.value = value
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
        self.value_opt = torch.optim.AdamW(self.value.parameters(), lr=lr)

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
        new_log_probs = gather_log_probs(log_probs, responses, prompt_lengths, response_lengths)
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

        logits, _ = self.policy(full, attention_mask=full_mask)
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
        self.policy_opt.zero_grad()
        policy_objective.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_opt.step()

        predicted_values = self.value(full, full_mask)
        target_values = rewards.to(predicted_values.device)
        value_loss = torch.nn.functional.mse_loss(predicted_values, target_values)
        self.value_opt.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
        self.value_opt.step()

        improvement = (predicted_values - old_values.to(predicted_values.device)).mean()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "kl": float(kl.item()),
            "entropy": float(entropy.item()),
            "value_improvement": float(improvement.item()),
        }


def train_ppo(*_args, **_kwargs) -> None:
    raise NotImplementedError(
        "The high level PPO training loop is intentionally left out to keep the repository minimal."
    )


def main() -> None:
    raise SystemExit("PPO training is intentionally left as an exercise in this minimal repo.")


if __name__ == "__main__":
    main()
