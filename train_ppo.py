import os
import random
import time
from typing import Optional

import torch
import torch.nn.functional as F

from data import PreferenceDataset
from gpt import GPT, GPTConfig
from train_rm import ScalarHead, train_reward_model
from simple_logger import TrainingLogger


def pad_responses(
    responses: list[torch.Tensor], pad: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not responses:
        return (
            torch.zeros(0, 0, dtype=torch.long),
            torch.zeros(0, 0, dtype=torch.float32),
        )
    max_len = max(r.size(0) for r in responses)
    out = torch.full((len(responses), max_len), pad, dtype=torch.long)
    mask = torch.zeros((len(responses), max_len), dtype=torch.float32)
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
    mask = torch.zeros_like(batch, dtype=torch.float32)
    response_mask = torch.zeros_like(batch, dtype=torch.float32)
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
    out = log_probs.new_zeros(batch, max_len)
    for i in range(batch):
        r_len = int(response_lengths[i].item())
        start = int(prompt_lengths[i].item())
        if r_len == 0:
            continue
        slice_log = log_probs[i, start : start + r_len]
        out[i, :r_len] = slice_log.gather(1, tokens[i, :r_len].unsqueeze(-1)).squeeze(-1)
    return out


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
        new_log_probs = gather_log_probs(
            log_probs,
            responses,
            prompt_lengths,
            response_lengths,
        )

        ratios = (new_log_probs - old_log_probs).exp()
        policy_targets = advantages.unsqueeze(1) * responses_mask
        unclipped = ratios * policy_targets
        clipped = torch.clamp(ratios, 1.0 - self.clip, 1.0 + self.clip) * policy_targets
        denom = responses_mask.sum().clamp(min=1.0)
        policy_loss = -torch.sum(torch.minimum(unclipped, clipped)) / denom

        probs = log_probs.exp()
        kl_per_token = (probs * (log_probs - ref_log_probs)).sum(dim=-1)
        kl = (kl_per_token * full_response_mask).sum() / full_response_mask.sum().clamp(min=1.0)

        entropy_per_token = -(probs * log_probs).sum(dim=-1)
        entropy = (entropy_per_token * full_response_mask).sum() / full_response_mask.sum().clamp(min=1.0)
        return policy_loss, kl, entropy

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
        advantages = batch["advantages"]

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

        rewards = batch["rewards"]
        values = self.value(full, full_mask)
        value_loss = F.mse_loss(values, rewards)

        self.policy_opt.zero_grad()
        total_policy = policy_loss + self.kl * kl - self.entropy * entropy
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


def train_ppo(
    preference_path: str,
    *,
    sft_path: Optional[str] = None,
    out_dir: str = "ppo_weights",
    dropout: float = 0.0,
    rm_batch: int = 4,
    rm_epochs: int = 1,
    rm_lr: float = 1e-5,
    ppo_batch: int = 4,
    ppo_epochs: int = 1,
    ppo_steps: int = 4,
    max_new: int = 64,
    ppo_lr: float = 1e-5,
    clip: float = 0.2,
    kl_coef: float = 0.1,
    entropy_coef: float = 0.01,
    device: Optional[torch.device] = None,
) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = GPTConfig(dropout=dropout)

    policy = GPT(config).to(device)
    reference = GPT(config).to(device)
    value_model = ScalarHead(config).to(device)
    reward_model = ScalarHead(config).to(device)

    if sft_path:
        state = torch.load(sft_path, map_location="cpu")
        policy.load_state_dict(state, strict=False)
        reference.load_state_dict(state, strict=False)
        value_model.body.load_state_dict(state, strict=False)
        reward_model.body.load_state_dict(state, strict=False)
    reference.load_state_dict(policy.state_dict())

    pref_data = PreferenceDataset(preference_path, block_size=config.block_size)
    reward_logger = TrainingLogger("reward_model")
    try:
        train_reward_model(
            reward_model,
            pref_data,
            batch_size=rm_batch,
            epochs=rm_epochs,
            lr=rm_lr,
            device=device,
            logger=reward_logger,
        )
    finally:
        reward_logger.close()

    bundle = pref_data.bundle
    trainer = PPOTrainer(
        policy,
        reference,
        value_model,
        reward_model,
        pad_token=bundle.pad,
        clip=clip,
        kl=kl_coef,
        entropy=entropy_coef,
        lr=ppo_lr,
    )

    prompts = [sample["prompt"] for sample in pref_data.samples]
    random.shuffle(prompts)
    batches = [
        prompts[i : i + ppo_batch]
        for i in range(0, len(prompts), ppo_batch)
    ]

    policy.train()
    ppo_logger = TrainingLogger("ppo")
    ppo_iteration = 0
    try:
        for epoch in range(ppo_epochs):
            epoch_iteration = 0
            for batch_prompts in batches:
                if not batch_prompts:
                    continue
                pad_len = max(p.size(0) for p in batch_prompts)
                prompt_tensor = torch.full((len(batch_prompts), pad_len), bundle.pad, dtype=torch.long)
                prompt_mask = torch.zeros_like(prompt_tensor, dtype=torch.float32)
                for i, prompt in enumerate(batch_prompts):
                    length = prompt.size(0)
                    prompt_tensor[i, :length] = prompt
                    prompt_mask[i, :length] = 1
                prompt_tensor = prompt_tensor.to(device)
                prompt_mask = prompt_mask.to(device)

                sampled = trainer.sample(prompt_tensor, prompt_mask, max_new)

                full = sampled["full"]
                mask = sampled["full_mask"]
                rewards = reward_model(full, mask)
                advantages = rewards - sampled["old_values"]
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
                sampled["advantages"] = advantages.detach()
                sampled["rewards"] = rewards.detach()

                for _ in range(ppo_steps):
                    start_time = time.perf_counter()
                    metrics = trainer.train_step(sampled)
                    elapsed = time.perf_counter() - start_time
                    ppo_iteration += 1
                    epoch_iteration += 1
                    ppo_logger.log(
                        {
                            "epoch": epoch + 1,
                            "epoch_iteration": epoch_iteration,
                            "iteration": ppo_iteration,
                            "policy_loss": metrics["policy_loss"],
                            "value_loss": metrics["value_loss"],
                            "kl": metrics["kl"],
                            "entropy": metrics["entropy"],
                            "reward": metrics["reward"],
                            "step_time": elapsed,
                        }
                    )

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"ppo_epoch_{epoch+1}.pt")
            torch.save(policy.state_dict(), path)
    finally:
        ppo_logger.close()


if __name__ == "__main__":
    PREFERENCE_PATH = "data/preferences_train.jsonl"
    SFT_PATH = None
    OUT_DIR = "ppo_weights"
    DROPOUT = 0.0
    RM_BATCH = 4
    RM_EPOCHS = 1
    RM_LR = 1e-5
    PPO_BATCH = 4
    PPO_EPOCHS = 1
    PPO_STEPS = 4
    MAX_NEW = 64
    PPO_LR = 1e-5
    CLIP = 0.2
    KL_COEF = 0.1
    ENTROPY_COEF = 0.01

    if not os.path.exists(PREFERENCE_PATH):
        raise FileNotFoundError(
            f"Preference data not found at {PREFERENCE_PATH}. Update the path before running."
        )

    train_ppo(
        PREFERENCE_PATH,
        sft_path=SFT_PATH,
        out_dir=OUT_DIR,
        dropout=DROPOUT,
        rm_batch=RM_BATCH,
        rm_epochs=RM_EPOCHS,
        rm_lr=RM_LR,
        ppo_batch=PPO_BATCH,
        ppo_epochs=PPO_EPOCHS,
        ppo_steps=PPO_STEPS,
        max_new=MAX_NEW,
        ppo_lr=PPO_LR,
        clip=CLIP,
        kl_coef=KL_COEF,
        entropy_coef=ENTROPY_COEF,
    )
