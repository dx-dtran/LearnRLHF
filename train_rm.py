"""Reward model utilities for RLHF training."""

from __future__ import annotations

import time
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import PreferenceDataset, collate_preferences
from gpt import GPT, GPTConfig

if TYPE_CHECKING:
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
