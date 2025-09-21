"""Reward model utilities for RLHF training."""

from __future__ import annotations

import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import PreferenceDataset, collate_preferences
from gpt import GPT, GPTConfig
from simple_logger import TrainingLogger


class ScalarHead(nn.Module):
    """A GPT backbone followed by a scalar prediction head."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.body = GPT(config)
        self.score = nn.Linear(config.n_embd, 1, bias=False)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return a scalar score for each sequence in ``tokens``.

        Parameters
        ----------
        tokens:
            Tokenized sequences padded to a common length.
        mask:
            Attention mask where ``1`` marks valid tokens and ``0`` marks padding.
        """

        hidden = self.body.transform(tokens, attention_mask=mask)
        lengths = mask.sum(dim=1).long() - 1
        last = hidden[torch.arange(hidden.size(0)), lengths.clamp(min=0)]
        return self.score(last).squeeze(-1)


def train_reward_model(
    model: ScalarHead,
    dataset: PreferenceDataset,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
    logger: Optional["TrainingLogger"] = None,
) -> None:
    """Fit ``model`` to human preference comparisons.

    The routine performs logistic regression on paired ``chosen``/``rejected``
    samples from ``dataset`` using the Bradley--Terry objective employed in the
    InstructGPT paper.
    """

    bundle = dataset.bundle
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_preferences(batch, bundle.pad),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    iteration = 0

    for epoch in range(epochs):
        epoch_iteration = 0
        for batch in loader:
            start_time = time.perf_counter()
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

            iteration += 1
            epoch_iteration += 1
            if logger is not None:
                elapsed = time.perf_counter() - start_time
                logger.log(
                    {
                        "epoch": epoch + 1,
                        "epoch_iteration": epoch_iteration,
                        "iteration": iteration,
                        "loss": loss.item(),
                        "step_time": elapsed,
                    }
                )


def run_reward_model_training(
    data_path: str,
    *,
    out_path: str,
    batch_size: int = 8,
    epochs: int = 1,
    lr: float = 1e-5,
    dropout: float = 0.0,
    init_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> str:
    """Train a scalar reward model on preference data and persist the weights."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Preference data not found at {data_path}. Prepare the JSONL file before training."
        )

    config = GPTConfig(dropout=dropout)
    model = ScalarHead(config)

    if init_path:
        state = torch.load(init_path, map_location="cpu")
        model.load_state_dict(state, strict=False)

    dataset = PreferenceDataset(data_path, block_size=config.block_size)
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    logger = TrainingLogger("rm")
    try:
        train_reward_model(
            model.to(device),
            dataset,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            device=device,
            logger=logger,
        )
    finally:
        logger.close()

    torch.save(model.state_dict(), out_path)
    return out_path


def main() -> None:
    # Edit the configuration below and run ``python train_rm.py`` to launch training.
    data_path = "data/hh_rlhf_preferences_train.jsonl"
    out_path = "weights/reward_model.pt"
    batch_size = 8
    epochs = 1
    lr = 1e-5
    dropout = 0.0
    init_path = None
    device_spec = None  # e.g. "cuda" or "cpu"; defaults to CUDA when available

    device = torch.device(device_spec) if device_spec else None
    run_reward_model_training(
        data_path,
        out_path=out_path,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        dropout=dropout,
        init_path=init_path,
        device=device,
    )


if __name__ == "__main__":
    main()
