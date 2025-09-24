"""Helper utilities for preparing batches in the PPO training loop."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def pad_token_sequences(
    sequences: Sequence[torch.Tensor], pad_token: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable length ``sequences`` with ``pad_token`` and build an attention mask."""

    if not sequences:
        empty = torch.zeros(0, 0, dtype=torch.long)
        empty_mask = torch.zeros(0, 0, dtype=torch.float32)
        return empty, empty_mask

    max_length = max(sequence.size(0) for sequence in sequences)
    batch_size = len(sequences)
    tokens = torch.full((batch_size, max_length), pad_token, dtype=torch.long)
    mask = torch.zeros((batch_size, max_length), dtype=torch.float32)

    for index, sequence in enumerate(sequences):
        length = sequence.size(0)
        if length:
            tokens[index, :length] = sequence
            mask[index, :length] = 1.0

    return tokens, mask


def batch_prompt_rows(
    rows: Sequence[dict[str, torch.Tensor]], pad_token: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate dataset rows that contain ``prompt`` tensors."""

    prompts = [row["prompt"] for row in rows]
    return pad_token_sequences(prompts, pad_token)


def gather_response_log_probs(
    log_probs: torch.Tensor,
    responses: torch.Tensor,
    prompt_lengths: torch.Tensor,
    response_lengths: torch.Tensor,
) -> torch.Tensor:
    """Pick log probabilities for the sampled responses."""

    batch_size, max_response_length = responses.shape
    collected = log_probs.new_zeros(batch_size, max_response_length)
    for row in range(batch_size):
        prompt_length = int(prompt_lengths[row].item())
        response_length = int(response_lengths[row].item())
        if response_length == 0:
            continue
        response_tokens = responses[row, :response_length]
        token_logits = log_probs[row, prompt_length : prompt_length + response_length]
        collected[row, :response_length] = token_logits.gather(
            1, response_tokens.unsqueeze(-1)
        ).squeeze(-1)
    return collected


def build_model_inputs(
    policy: torch.nn.Module,
    value_head: torch.nn.Module,
    prompts: torch.Tensor,
    prompt_mask: torch.Tensor,
    *,
    max_new_tokens: int,
    pad_token: int,
    eos_token: int,
) -> dict[str, torch.Tensor]:
    """Generate responses and compute cached statistics for PPO."""

    device = prompts.device
    prompt_lengths = prompt_mask.sum(dim=1).long().tolist()
    prompt_sequences = [prompts[index, : length].cpu() for index, length in enumerate(prompt_lengths)]

    responses: list[torch.Tensor] = []
    full_sequences: list[torch.Tensor] = []
    response_lengths: list[int] = []

    block_size = policy.config.block_size
    for prompt_sequence, prompt_length in zip(prompt_sequences, prompt_lengths):
        available_context = max(block_size - prompt_length, 0)
        allowed_new_tokens = min(max_new_tokens, available_context)
        if allowed_new_tokens <= 0:
            full_sequence = prompt_sequence.clone()
            response = prompt_sequence.new_empty((0,), dtype=torch.long)
        else:
            generated = policy.generate(
                prompt_sequence.unsqueeze(0).to(device), allowed_new_tokens, eos_token=eos_token
            )
            full_sequence = generated[0].detach().cpu()
            response = full_sequence[prompt_length:]
        full_sequences.append(full_sequence)
        responses.append(response)
        response_lengths.append(response.size(0))

    full_tokens, full_mask = pad_token_sequences(full_sequences, pad_token)
    response_tokens, response_mask = pad_token_sequences(responses, pad_token)

    response_token_mask = torch.zeros_like(full_mask)
    for index, (prompt_length, response_length) in enumerate(zip(prompt_lengths, response_lengths)):
        if response_length:
            response_token_mask[index, prompt_length : prompt_length + response_length] = 1.0

    full_tokens = full_tokens.to(device)
    full_mask = full_mask.to(device)
    response_tokens = response_tokens.to(device)
    response_mask = response_mask.to(device)
    response_token_mask = response_token_mask.to(device)
    prompt_lengths_tensor = torch.tensor(prompt_lengths, device=device, dtype=torch.long)
    response_lengths_tensor = torch.tensor(response_lengths, device=device, dtype=torch.long)

    with torch.no_grad():
        hidden = policy.transform(full_tokens, attention_mask=full_mask)
        logits = policy.head(hidden)
        log_probs = F.log_softmax(logits, dim=-1)
        old_log_probs = gather_response_log_probs(
            log_probs, response_tokens, prompt_lengths_tensor, response_lengths_tensor
        )
        old_values = value_head(hidden, full_mask)

    return {
        "full": full_tokens,
        "full_mask": full_mask,
        "responses": response_tokens,
        "responses_mask": response_mask,
        "prompt_lengths": prompt_lengths_tensor,
        "response_lengths": response_lengths_tensor,
        "old_log_probs": old_log_probs.detach(),
        "old_values": old_values.detach(),
        "response_mask": response_token_mask,
    }
