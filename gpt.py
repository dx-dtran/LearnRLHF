import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import MutableMapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0



class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        if self.head_dim * config.n_head != config.n_embd:
            raise ValueError("n_embd must be divisible by n_head")

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        no_valid_keys: Optional[torch.Tensor] = None
        if mask is not None:
            att = att.masked_fill(~mask, float("-inf"))
            no_valid_keys = ~mask.any(dim=-1, keepdim=True)
            att = torch.where(no_valid_keys, torch.zeros_like(att), att)

        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        if query_mask is not None:
            query_mask_float = query_mask.to(dtype=att.dtype)
            att = att * query_mask_float
        else:
            query_mask_float = None

        out = att @ v

        if query_mask_float is not None:
            out = out * query_mask_float
        if no_valid_keys is not None:
            out = out.masked_fill(no_valid_keys, 0.0)

        out = out.transpose(1, 2).contiguous().view(b, t, c)
        out = self.c_proj(out)
        return self.resid_dropout(out)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=True)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask, query_mask)
        x = x + self.resid_dropout(self.mlp(self.ln2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embed = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = self.token_embed.weight
        self.drop = nn.Dropout(config.dropout)

    def _build_mask(
        self, attention_mask: torch.Tensor | None, length: int, device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        causal = torch.tril(torch.ones(length, length, device=device, dtype=torch.bool))
        causal = causal.view(1, 1, length, length)
        if attention_mask is None:
            return causal, None
        attn = attention_mask.to(device=device, dtype=torch.bool)
        key_mask = attn[:, None, None, :]
        query_mask = attn[:, None, :, None]
        full_mask = causal & key_mask
        return full_mask, query_mask

    def transform(self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t = tokens.shape
        if t > self.config.block_size:
            raise ValueError("input sequence is longer than block_size")
        pos = torch.arange(t, device=tokens.device)
        x = self.token_embed(tokens) + self.position_embed(pos)[None, :, :]
        x = self.drop(x)
        mask, query_mask = self._build_mask(attention_mask, t, tokens.device)
        for block in self.blocks:
            x = block(x, mask, query_mask)
        return self.ln(x)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self.transform(tokens, attention_mask)
        logits = self.head(hidden)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        eos_token: int | None = None,
    ) -> torch.Tensor:
        sequence = tokens
        for _ in range(max_new_tokens):
            window = sequence[:, -self.config.block_size :]
            logits, _ = self.forward(window)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            sequence = torch.cat([sequence, next_token], dim=1)
            if eos_token is not None and torch.any(next_token == eos_token):
                break
        return sequence


_TRANSPOSE_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def maybe_transpose_gpt_state_dict(
    state_dict: MutableMapping[str, torch.Tensor],
) -> MutableMapping[str, torch.Tensor]:
    """Transpose legacy GPT-2 weight matrices to match the current module layout."""

    needs_transpose = False
    for key, value in state_dict.items():
        if value.ndim != 2:
            continue
        if any(key.endswith(suffix) for suffix in _TRANSPOSE_SUFFIXES):
            needs_transpose = True
            break

    if not needs_transpose:
        return state_dict

    if isinstance(state_dict, OrderedDict):
        items = state_dict.items()
        new_state = OrderedDict()
    else:
        items = state_dict.items()
        new_state = {}

    for key, value in items:
        if value.ndim == 2 and any(key.endswith(suffix) for suffix in _TRANSPOSE_SUFFIXES):
            new_state[key] = value.transpose(0, 1)
        else:
            new_state[key] = value
    return new_state
