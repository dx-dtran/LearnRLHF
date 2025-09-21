import math
from dataclasses import dataclass

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

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        out = att @ v

        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
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

    def _build_mask(
        self, attention_mask: torch.Tensor | None, length: int, device: torch.device
    ) -> torch.Tensor:
        causal = torch.tril(torch.ones(length, length, device=device, dtype=torch.bool))
        causal = causal.view(1, 1, length, length)
        if attention_mask is None:
            return causal
        attn = attention_mask.to(device=device, dtype=torch.bool)
        attn = attn[:, None, None, :]
        query_mask = attention_mask.to(device=device, dtype=torch.bool)[:, None, :, None]
        full_mask = causal & attn & query_mask
        return full_mask

    def transform(self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t = tokens.shape
        if t > self.config.block_size:
            raise ValueError("input sequence is longer than block_size")
        pos = torch.arange(t, device=tokens.device)
        x = self.token_embed(tokens) + self.position_embed(pos)[None, :, :]
        mask = self._build_mask(attention_mask, t, tokens.device)
        for block in self.blocks:
            x = block(x, mask)
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
