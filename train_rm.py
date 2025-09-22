import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import (
    TokenizerBundle,
    PreferenceDataset,
    build_tokenizer,
    collate_preferences,
)
from gpt import GPT, GPTConfig, load_gpt_checkpoint


class ScalarHead(nn.Module):
    """A GPT backbone followed by a scalar head used for reward/value models."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.body = GPT(config)
        self.score = nn.Linear(config.n_embd, 1, bias=False)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.body.transform(tokens, attention_mask=mask)
        lengths = mask.sum(dim=1).long().clamp(min=1) - 1
        last_token = hidden[torch.arange(hidden.size(0)), lengths]
        return self.score(last_token).squeeze(-1)


def preference_loss(chosen: torch.Tensor, rejected: torch.Tensor) -> torch.Tensor:
    return -torch.nn.functional.logsigmoid(chosen - rejected).mean()


def _resolve_initial_state(
    init_checkpoint: Optional[str],
    bundle: TokenizerBundle,
    block_size: int,
    dropout: float,
) -> tuple[GPTConfig, dict | None]:
    config = GPTConfig(
        vocab_size=bundle.encoder.n_vocab,
        block_size=block_size,
        dropout=dropout,
    )
    state = None
    if init_checkpoint:
        if not os.path.exists(init_checkpoint):
            raise FileNotFoundError(
                f"Initial checkpoint {init_checkpoint} does not exist; provide a Torch state dict"
            )
        base_config, base_state = load_gpt_checkpoint(init_checkpoint)
        if base_config.vocab_size != bundle.encoder.n_vocab:
            raise ValueError(
                "Tokenizer vocabulary does not match the checkpoint embedding size"
            )
        if block_size > base_config.block_size:
            raise ValueError(
                "Requested block_size exceeds the positional embeddings in the checkpoint"
            )
        base_config.block_size = min(block_size, base_config.block_size)
        base_config.dropout = dropout
        remapped_state: dict[str, torch.Tensor] = {}
        for key, value in base_state.items():
            if key.startswith(("token_embed", "position_embed", "blocks", "ln")):
                remapped_state[f"body.{key}"] = value
        state = remapped_state
        config = base_config
    return config, state


def train_reward_model(
    data_path: str,
    *,
    out_path: str,
    batch_size: int = 12,
    epochs: int = 3,
    lr: float = 1e-5,
    dropout: float = 0.0,
    grad_accumulation_steps: int = 1,
    init_path: Optional[str] = None,
    block_size: int = 1024,
    device: Optional[torch.device] = None,
) -> str:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Preference data not found at {data_path}. Prepare the JSONL file before training."
        )

    bundle = build_tokenizer()
    config, state = _resolve_initial_state(init_path, bundle, block_size, dropout)
    model = ScalarHead(config)

    if state is not None:
        model.load_state_dict(state, strict=False)

    dataset = PreferenceDataset(data_path, block_size=config.block_size, bundle=bundle)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_preferences(batch, bundle.pad),
    )

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    if grad_accumulation_steps < 1:
        raise ValueError("grad_accumulation_steps must be a positive integer")

    for epoch_idx in range(epochs):
        optimizer.zero_grad()
        step_in_epoch = 0
        running_loss = 0.0
        preference_correct = 0
        preference_total = 0
        for batch in loader:
            chosen = batch["chosen"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected = batch["rejected"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)

            chosen_scores = model(chosen, chosen_mask)
            rejected_scores = model(rejected, rejected_mask)
            loss = preference_loss(chosen_scores, rejected_scores)

            with torch.no_grad():
                preference_correct += (chosen_scores > rejected_scores).sum().item()
                preference_total += chosen_scores.size(0)
                running_loss += loss.item()

            (loss / grad_accumulation_steps).backward()
            step_in_epoch += 1

            if step_in_epoch % grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        remainder = step_in_epoch % grad_accumulation_steps
        if remainder != 0:
            scale = grad_accumulation_steps / remainder
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.mul_(scale)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        average_loss = running_loss / max(step_in_epoch, 1)
        accuracy = preference_correct / max(preference_total, 1)
        print(
            f"Epoch {epoch_idx + 1}/{epochs}: loss={average_loss:.4f}, "
            f"preference_accuracy={accuracy:.4f}"
        )

    torch.save(model.state_dict(), out_path)
    return out_path


def main() -> None:
    data_path = "data/hh_rlhf_rm_train.jsonl"
    out_path = "weights/rm.pt"
    batch_size = 12
    epochs = 3
    lr = 1e-5
    dropout = 0.0
    grad_accumulation_steps = 1
    init_path: Optional[str] = None
    block_size = 1024
    device = None

    device_obj = torch.device(device) if device else None
    train_reward_model(
        data_path,
        out_path=out_path,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        dropout=dropout,
        grad_accumulation_steps=grad_accumulation_steps,
        init_path=init_path,
        block_size=block_size,
        device=device_obj,
    )


if __name__ == "__main__":
    main()
