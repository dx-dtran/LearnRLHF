import os
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import PreferenceDataset, collate_preferences
from gpt import GPT, GPTConfig


class ScalarHead(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.body = GPT(config)
        self.score = nn.Linear(config.n_embd, 1, bias=False)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.body.transform(tokens, attention_mask=mask)
        lengths = mask.sum(dim=1) - 1
        last = hidden[torch.arange(hidden.size(0)), lengths.clamp(min=0)]
        return self.score(last).squeeze(-1)


def pad_responses(responses: list[torch.Tensor], pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not responses:
        return torch.zeros(0, 0, dtype=torch.long), torch.zeros(0, 0, dtype=torch.long)
    max_len = max(r.size(0) for r in responses)
    out = torch.full((len(responses), max_len), pad, dtype=torch.long)
    mask = torch.zeros((len(responses), max_len), dtype=torch.long)
    for i, resp in enumerate(responses):
        length = resp.size(0)
        if length:
            out[i, :length] = resp
            mask[i, :length] = 1
    return out, mask


def build_full_sequences(
    prompts: list[torch.Tensor],
    responses: list[torch.Tensor],
    pad: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    full_lengths = [p.size(0) + r.size(0) for p, r in zip(prompts, responses)]
    max_full = max(full_lengths) if full_lengths else 0
    batch = torch.full((len(prompts), max_full), pad, dtype=torch.long)
    mask = torch.zeros_like(batch)
    response_mask = torch.zeros_like(batch)
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        p_len = prompt.size(0)
        r_len = response.size(0)
        batch[i, :p_len] = prompt
        mask[i, :p_len] = 1
        if r_len:
            batch[i, p_len : p_len + r_len] = response
            mask[i, p_len : p_len + r_len] = 1
            response_mask[i, p_len : p_len + r_len] = 1
    return batch, mask, response_mask


def gather_log_probs(
    log_probs: torch.Tensor,
    tokens: torch.Tensor,
    prompt_lengths: torch.Tensor,
    response_lengths: torch.Tensor,
) -> torch.Tensor:
    batch, max_len = tokens.shape
    out = torch.zeros(batch, max_len, device=log_probs.device)
    for i in range(batch):
        r_len = int(response_lengths[i].item())
        start = int(prompt_lengths[i].item())
        if r_len == 0:
            continue
        slice_log = log_probs[i, start : start + r_len]
        out[i, :r_len] = slice_log.gather(1, tokens[i, :r_len].unsqueeze(-1)).squeeze(-1)
    return out


def ppo_policy_objective(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    responses: torch.Tensor,
    responses_mask: torch.Tensor,
    prompt_lengths: torch.Tensor,
    response_lengths: torch.Tensor,
    response_token_mask: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip: float,
    kl_coeff: float,
    entropy_coeff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    responses_mask = responses_mask.to(log_probs.dtype)
    response_token_mask = response_token_mask.to(log_probs.dtype)
    advantages = advantages.to(log_probs.dtype)

    new_log_probs = gather_log_probs(
        log_probs,
        responses,
        prompt_lengths,
        response_lengths,
    )

    ratios = (new_log_probs - old_log_probs).exp()
    policy_targets = advantages.unsqueeze(1) * responses_mask
    unclipped = ratios * policy_targets
    clipped = torch.clamp(ratios, 1.0 - clip, 1.0 + clip) * policy_targets
    denom = responses_mask.sum().clamp(min=1).float()
    policy_loss = -torch.sum(torch.minimum(unclipped, clipped)) / denom

    probs = log_probs.exp()
    kl_per_token = (probs * (log_probs - ref_log_probs)).sum(dim=-1)
    mask_norm = response_token_mask.sum().clamp(min=1).float()
    kl = (kl_per_token * response_token_mask).sum() / mask_norm

    entropy_per_token = -(probs * log_probs).sum(dim=-1)
    entropy = (entropy_per_token * response_token_mask).sum() / mask_norm

    total_policy = policy_loss + kl_coeff * kl - entropy_coeff * entropy
    return total_policy, policy_loss, kl, entropy


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
    ):
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

    def sample(self, prompts: torch.Tensor, mask: torch.Tensor, max_new: int) -> dict[str, torch.Tensor]:
        device = prompts.device
        prompt_lengths = mask.sum(dim=1)
        sequences = []
        responses = []
        for i in range(prompts.size(0)):
            length = int(prompt_lengths[i].item())
            seq = prompts[i, :length].unsqueeze(0)
            seq = seq.to(device)
            generated = self.policy.generate(seq, max_new, eos_token=self.pad)
            full = generated[0]
            response = full[length:]
            sequences.append(full.cpu())
            responses.append(response.cpu())

        response_pad, response_mask = pad_responses(responses, self.pad)
        prompt_list = [prompts[i, : int(prompt_lengths[i].item())].cpu() for i in range(prompts.size(0))]
        full, full_mask, response_token_mask = build_full_sequences(prompt_list, responses, self.pad)

        prompt_lengths = torch.tensor([p.size(0) for p in prompt_list], device=device)
        response_lengths = torch.tensor([r.size(0) for r in responses], device=device)

        full = full.to(device)
        full_mask = full_mask.to(device)
        response_token_mask = response_token_mask.to(device)
        response_pad = response_pad.to(device)
        response_mask = response_mask.to(device)

        with torch.no_grad():
            logits, _ = self.policy(full, attention_mask=full_mask)
            log_probs = F.log_softmax(logits, dim=-1)
            old_log = gather_log_probs(
                log_probs,
                response_pad,
                prompt_lengths,
                response_lengths,
            )
            values = self.value(full, full_mask)

        return {
            "full": full,
            "full_mask": full_mask,
            "response_mask": response_token_mask,
            "responses": response_pad,
            "responses_mask": response_mask,
            "prompt_lengths": prompt_lengths,
            "response_lengths": response_lengths,
            "old_log_probs": old_log,
            "old_values": values,
        }

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        full = batch["full"]
        full_mask = batch["full_mask"]
        response_mask = batch["response_mask"]
        responses = batch["responses"]
        responses_mask = batch["responses_mask"]
        prompt_lengths = batch["prompt_lengths"]
        response_lengths = batch["response_lengths"]
        old_log_probs = batch["old_log_probs"]
        old_values = batch["old_values"]

        logits, _ = self.policy(full, attention_mask=full_mask)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            ref_logits, _ = self.reference(full, attention_mask=full_mask)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        total_policy, policy_loss, kl, entropy = ppo_policy_objective(
            log_probs,
            ref_log_probs,
            responses,
            responses_mask,
            prompt_lengths,
            response_lengths,
            response_mask,
            old_log_probs,
            batch["advantages"],
            self.clip,
            self.kl,
            self.entropy,
        )

        rewards = batch["rewards"]
        values = self.value(full, full_mask)
        value_loss = F.mse_loss(values, rewards)

        self.policy_opt.zero_grad()
        total_policy.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_opt.step()

        self.value_opt.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
        self.value_opt.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "kl": kl.item(),
            "entropy": entropy.item(),
            "reward": rewards.mean().item(),
        }


def train_reward_model(
    model: ScalarHead,
    dataset: PreferenceDataset,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
):
    bundle = dataset.bundle
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_preferences(batch, bundle.pad),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            chosen = batch["chosen"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected = batch["rejected"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)

            chosen_scores = model(chosen, chosen_mask)
            rejected_scores = model(rejected, rejected_mask)
            loss = -torch.nn.functional.logsigmoid(chosen_scores - rejected_scores).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


@dataclass
class PPOConfig:
    preference_path: str
    sft_path: Optional[str] = None
    out_dir: Optional[str] = "ppo_weights"
    dropout: float = 0.0
    rm_batch: int = 4
    rm_epochs: int = 1
    rm_lr: float = 1e-5
    ppo_batch: int = 4
    ppo_epochs: int = 1
    ppo_steps: int = 4
    max_new: int = 64
    ppo_lr: float = 1e-5
    clip: float = 0.2
    kl: float = 0.1
    entropy: float = 0.01
    device: Optional[str] = None


def run_rlhf(config: PPOConfig) -> None:
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_config = GPTConfig(dropout=config.dropout)

    policy = GPT(model_config).to(device)
    reference = GPT(model_config).to(device)
    value_model = ScalarHead(model_config).to(device)
    reward_model = ScalarHead(model_config).to(device)

    if config.sft_path:
        state = torch.load(config.sft_path, map_location="cpu")
        policy.load_state_dict(state, strict=False)
        reference.load_state_dict(state, strict=False)
        value_model.body.load_state_dict(state, strict=False)
        reward_model.body.load_state_dict(state, strict=False)
    reference.load_state_dict(policy.state_dict())

    pref_data = PreferenceDataset(config.preference_path, block_size=model_config.block_size)
    train_reward_model(
        reward_model,
        pref_data,
        batch_size=config.rm_batch,
        epochs=config.rm_epochs,
        lr=config.rm_lr,
        device=device,
    )

    bundle = pref_data.bundle
    trainer = PPOTrainer(
        policy,
        reference,
        value_model,
        reward_model,
        pad_token=bundle.pad,
        clip=config.clip,
        kl=config.kl,
        entropy=config.entropy,
        lr=config.ppo_lr,
    )

    prompts = [sample["prompt"] for sample in pref_data.samples]
    random.shuffle(prompts)
    batches = [
        prompts[i : i + config.ppo_batch]
        for i in range(0, len(prompts), config.ppo_batch)
    ]

    policy.train()
    for epoch in range(config.ppo_epochs):
        for batch_prompts in batches:
            if not batch_prompts:
                continue
            pad_len = max(p.size(0) for p in batch_prompts)
            prompt_tensor = torch.full((len(batch_prompts), pad_len), bundle.pad, dtype=torch.long)
            prompt_mask = torch.zeros_like(prompt_tensor)
            for i, prompt in enumerate(batch_prompts):
                length = prompt.size(0)
                prompt_tensor[i, :length] = prompt
                prompt_mask[i, :length] = 1
            prompt_tensor = prompt_tensor.to(device)
            prompt_mask = prompt_mask.to(device)

            sampled = trainer.sample(prompt_tensor, prompt_mask, config.max_new)

            full = sampled["full"]
            mask = sampled["full_mask"]
            rewards = reward_model(full, mask)
            advantages = rewards - sampled["old_values"]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
            sampled["advantages"] = advantages.detach()
            sampled["rewards"] = rewards.detach()

            for _ in range(config.ppo_steps):
                trainer.train_step(sampled)

        if config.out_dir:
            os.makedirs(config.out_dir, exist_ok=True)
            path = os.path.join(config.out_dir, f"ppo_epoch_{epoch + 1}.pt")
            torch.save(policy.state_dict(), path)


if __name__ == "__main__":
    RUN = PPOConfig(
        preference_path="/path/to/anthropic_hh_preference.jsonl",
        sft_path=None,
        out_dir="ppo_weights",
        dropout=0.0,
        rm_batch=4,
        rm_epochs=1,
        rm_lr=1e-5,
        ppo_batch=4,
        ppo_epochs=1,
        ppo_steps=4,
        max_new=64,
        ppo_lr=1e-5,
        clip=0.2,
        kl=0.1,
        entropy=0.01,
    )
    run_rlhf(RUN)
