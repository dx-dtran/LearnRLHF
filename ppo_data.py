"""Utilities for preparing PPO training batches."""

from __future__ import annotations

from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from gpt import GPT


def pad_sequences(sequences: Iterable[torch.Tensor], pad_token: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad ``sequences`` to a common length returning both the tokens and mask."""

    sequences = list(sequences)
    if not sequences:
        empty = torch.zeros(0, 0, dtype=torch.long)
        return empty, empty.to(dtype=torch.float32)

    max_len = max(seq.size(0) for seq in sequences)
    tokens = torch.full((len(sequences), max_len), pad_token, dtype=torch.long)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.float32)
    for idx, seq in enumerate(sequences):
        length = seq.size(0)
        if length:
            tokens[idx, :length] = seq
            mask[idx, :length] = 1.0
    return tokens, mask


def batch_prompts(rows: list[dict[str, torch.Tensor]], pad_token: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch prompt tensors from ``rows`` using ``pad_token`` for padding."""

    prompts = [row["prompt"] for row in rows]
    return pad_sequences(prompts, pad_token)


def collect_response_log_probs(
    log_probs: torch.Tensor,
    responses: torch.Tensor,
    prompt_lengths: torch.Tensor,
    response_lengths: torch.Tensor,
) -> torch.Tensor:
    """Gather per-token log probabilities for each sampled response."""

    batch_size, max_response = responses.shape
    collected = log_probs.new_zeros(batch_size, max_response)
    for idx in range(batch_size):
        start = int(prompt_lengths[idx].item())
        length = int(response_lengths[idx].item())
        if length == 0:
            continue
        tokens = responses[idx, :length]
        window = log_probs[idx, start : start + length]
        collected[idx, :length] = window.gather(1, tokens.unsqueeze(-1)).squeeze(-1)
    return collected


def prepare_sample_batch(
    prompts: torch.Tensor,
    prompt_mask: torch.Tensor,
    *,
    policy: GPT,
    value_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pad_token: int,
    eos_token: int,
    max_new_tokens: int,
) -> dict[str, torch.Tensor]:
    """Expand ``prompts`` with generated responses and compute cached values."""

    device = prompts.device
    prompt_lengths = prompt_mask.sum(dim=1).long().tolist()
    prompt_sequences = [prompts[idx, : length].cpu() for idx, length in enumerate(prompt_lengths)]

    responses: list[torch.Tensor] = []
    full_sequences: list[torch.Tensor] = []
    response_lengths: list[int] = []
    block_size = policy.config.block_size
    for prompt_seq, length in zip(prompt_sequences, prompt_lengths):
        headroom = max(block_size - length, 0)
        allowed = min(max_new_tokens, headroom)
        if allowed <= 0:
            full_sequence = prompt_seq.clone()
            response = prompt_seq.new_empty((0,), dtype=torch.long)
        else:
            generated = policy.generate(
                prompt_seq.unsqueeze(0).to(device),
                allowed,
                eos_token=eos_token,
            )
            full_sequence = generated[0].detach().cpu()
            response = full_sequence[length:]
        full_sequences.append(full_sequence)
        responses.append(response)
        response_lengths.append(response.size(0))

    full_tokens, full_mask = pad_sequences(full_sequences, pad_token)
    response_tokens, response_padding_mask = pad_sequences(responses, pad_token)

    response_token_mask = torch.zeros_like(full_mask)
    for idx, (prompt_len, response_len) in enumerate(zip(prompt_lengths, response_lengths)):
        if response_len:
            response_token_mask[idx, prompt_len : prompt_len + response_len] = 1.0

    full_tokens = full_tokens.to(device)
    full_mask = full_mask.to(device)
    response_tokens = response_tokens.to(device)
    response_padding_mask = response_padding_mask.to(device)
    response_token_mask = response_token_mask.to(device)

    prompt_lengths_tensor = torch.tensor(prompt_lengths, device=device, dtype=torch.long)
    response_lengths_tensor = torch.tensor(response_lengths, device=device, dtype=torch.long)

    with torch.no_grad():
        hidden = policy.transform(full_tokens, attention_mask=full_mask)
        logits = policy.head(hidden)
        log_probs = F.log_softmax(logits, dim=-1)
        old_log_probs = collect_response_log_probs(
            log_probs,
            response_tokens,
            prompt_lengths_tensor,
            response_lengths_tensor,
        )
        old_values = value_fn(hidden, full_mask)

    return {
        "full": full_tokens,
        "full_mask": full_mask,
        "responses": response_tokens,
        "responses_mask": response_padding_mask,
        "prompt_lengths": prompt_lengths_tensor,
        "response_lengths": response_lengths_tensor,
        "old_log_probs": old_log_probs.detach(),
        "old_values": old_values.detach(),
        "response_token_mask": response_token_mask,
    }
