from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

from data import PreferenceDataset
from gpt import GPT, GPTConfig
from train_rm import ScalarHead


def _pad_batch(seqs: list[torch.Tensor], pad_token: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not seqs:
        return (
            torch.zeros(0, 0, dtype=torch.long),
            torch.zeros(0, 0, dtype=torch.float32),
        )
    max_len = max(seq.size(0) for seq in seqs)
    tokens = torch.full((len(seqs), max_len), pad_token, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.float32)
    for i, seq in enumerate(seqs):
        length = seq.size(0)
        if length:
            tokens[i, :length] = seq
            mask[i, :length] = 1.0
    return tokens, mask


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
            "reward": float(rewards.mean().item()),
        }


def _prepare_sample_batch(
    prompts: torch.Tensor,
    prompt_mask: torch.Tensor,
    trainer: PPOTrainer,
    max_new_tokens: int,
    pad_token: int,
    eos_token: int,
) -> dict[str, torch.Tensor]:
    device = prompts.device

    prompt_lengths = prompt_mask.sum(dim=1).long().tolist()
    prompt_list = [prompts[i, : length].cpu() for i, length in enumerate(prompt_lengths)]

    responses: list[torch.Tensor] = []
    full_sequences: list[torch.Tensor] = []
    response_lengths: list[int] = []
    block_size = trainer.policy.config.block_size
    for prompt, length in zip(prompt_list, prompt_lengths):
        headroom = max(block_size - length, 0)
        allowed_new_tokens = min(max_new_tokens, headroom)
        if allowed_new_tokens <= 0:
            full = prompt.clone()
            response = prompt.new_empty((0,), dtype=torch.long)
        else:
            generated = trainer.policy.generate(
                prompt.unsqueeze(0).to(device), allowed_new_tokens, eos_token=eos_token
            )
            full = generated[0].detach().cpu()
            response = full[length:]
        full_sequences.append(full)
        responses.append(response)
        response_lengths.append(response.size(0))

    full_tokens, full_mask = _pad_batch(full_sequences, pad_token)
    responses_tokens, responses_mask = _pad_batch(responses, pad_token)

    response_token_mask = torch.zeros_like(full_mask)
    for i, (p_len, r_len) in enumerate(zip(prompt_lengths, response_lengths)):
        if r_len:
            response_token_mask[i, p_len : p_len + r_len] = 1.0

    full_tokens = full_tokens.to(device)
    full_mask = full_mask.to(device)
    responses_tokens = responses_tokens.to(device)
    responses_mask = responses_mask.to(device)
    response_token_mask = response_token_mask.to(device)
    prompt_lengths_tensor = torch.tensor(prompt_lengths, device=device, dtype=torch.long)
    response_lengths_tensor = torch.tensor(response_lengths, device=device, dtype=torch.long)

    with torch.no_grad():
        logits, _ = trainer.policy(full_tokens, attention_mask=full_mask)
        log_probs = F.log_softmax(logits, dim=-1)
        old_log_probs = gather_log_probs(
            log_probs,
            responses_tokens,
            prompt_lengths_tensor,
            response_lengths_tensor,
        )
        old_values = trainer.value(full_tokens, full_mask)

    return {
        "full": full_tokens,
        "full_mask": full_mask,
        "responses": responses_tokens,
        "responses_mask": responses_mask,
        "prompt_lengths": prompt_lengths_tensor,
        "response_lengths": response_lengths_tensor,
        "old_log_probs": old_log_probs.detach(),
        "old_values": old_values.detach(),
        "response_mask": response_token_mask,
    }


def _batch_prompts(rows: list[dict[str, torch.Tensor]], pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    prompts = [row["prompt"] for row in rows]
    tokens, mask = _pad_batch(prompts, pad)
    return tokens, mask


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
    policy = GPT(config)
    reference = GPT(config)
    value_model = ScalarHead(config)
    reward_model = ScalarHead(config)

    if policy_init:
        state = torch.load(policy_init, map_location="cpu")
        policy.load_state_dict(state, strict=False)
        reference.load_state_dict(state, strict=False)
        value_model.body.load_state_dict(state, strict=False)

    reference.load_state_dict(policy.state_dict())
    value_model.body.load_state_dict(policy.state_dict(), strict=False)
    reward_state = torch.load(reward_path, map_location="cpu")
    reward_model.load_state_dict(reward_state, strict=False)

    policy.to(device)
    reference.to(device)
    value_model.to(device)
    reward_model.to(device)

    dataset = PreferenceDataset(preference_path, block_size=config.block_size)
    bundle = dataset.bundle

    trainer = PPOTrainer(
        policy,
        reference,
        value_model,
        reward_model,
        pad_token=bundle.pad,
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
            prompt_tokens, prompt_mask = _batch_prompts(rows, bundle.pad)
            prompt_tokens = prompt_tokens.to(device)
            prompt_mask = prompt_mask.to(device)

            sample = _prepare_sample_batch(
                prompt_tokens,
                prompt_mask,
                trainer,
                max_new_tokens,
                bundle.pad,
                bundle.eos,
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
    policy_init = None
    out_path = "weights/ppo_policy.pt"
    batch_size = 4
    epochs = 1
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
