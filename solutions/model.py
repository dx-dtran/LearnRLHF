"""
model.py — GPT-2 from scratch, nanoGPT style.

Part 1 of the crash course. Most of this file is given. Your job:

    [FILL 1.1]  CausalSelfAttention.forward  — the attention math itself
    [FILL 1.2]  GPT.forward_hidden           — assemble embeddings + blocks + ln_f

Everything else (MLP, Block, weight loading, sampling) is provided so you can get to
the RLHF parts quickly. Read the provided code anyway; you will reuse pieces of it.

A note on padding and positions, used everywhere downstream:
    - `attention_mask` is (B, T) with 1 = real token, 0 = pad. Pad columns are hidden
      from attention as keys.
    - Position ids are derived from the attention mask: a token's position is the
      number of real tokens before it. With right padding this is just 0..T-1; with
      LEFT padding (PPO prompt batches) it keeps absolute positions consistent.
    - A fully-masked attention row (a pad query) would softmax over all -inf and
      produce NaN. We therefore mask pads with a large-but-finite negative number, so
      pad rows produce garbage-but-finite outputs that downstream losses mask away.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


def positions_from_mask(
    idx: torch.Tensor, attention_mask: torch.Tensor | None
) -> torch.Tensor:
    """Position ids = number of real tokens strictly before each position."""
    if attention_mask is None:
        return torch.arange(idx.size(1), device=idx.device).unsqueeze(0)
    pos = attention_mask.long().cumsum(dim=1) - 1
    return pos.clamp_min(0)


# =====================================================================================
# LayerNorm by hand (provided — you wrote this in an earlier pass; kept as reference
# for how the rest of the from-scratch modules should look)
# =====================================================================================


class ManualLayerNorm(nn.Module):
    """
    LayerNorm with affine params, no nn.LayerNorm under the hood.

    GPT-2 uses biased variance (divide by C, not C-1) and eps=1e-5, matching
    torch.nn.LayerNorm.
    """

    def __init__(self, n_embd: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        variance = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
        x_hat = (x - mu) / torch.sqrt(variance + self.eps)
        return self.weight * x_hat + self.bias


# =====================================================================================
# [FILL 1.1] — Causal self-attention
# =====================================================================================


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Parameters (names match GPT-2 so weight loading is a transpose away):
        c_attn : Linear(n_embd, 3 * n_embd, bias=True)    # fused Q,K,V
        c_proj : Linear(n_embd, n_embd, bias=True)        # output projection

    Forward recipe:
        qkv = c_attn(x)                                   # (B, T, 3C)
        q, k, v = split last dim into thirds, each reshaped to (B, nh, T, hs)
        att = (q @ k^T) / sqrt(hs)                        # (B, nh, T, T)
        causal mask: position t may attend to positions <= t only
        key-padding mask: if attention_mask is given, pad columns get a large
            negative fill (use torch.finfo(att.dtype).min, NOT -inf — a pad query row
            would otherwise be all -inf and softmax to NaN)
        p = softmax(att, dim=-1)
        y = p @ v                                         # (B, nh, T, hs)
        reshape back to (B, T, C), then c_proj             <-- don't forget c_proj!

    Implement with explicit matmuls first. After test_attention_* pass, you may swap
    in F.scaled_dot_product_attention for speed if you want; keep the explicit
    version in a comment.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)

        # causal mask buffer, slice with [:T, :T]
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(self.block_size, self.block_size)).bool(),
            persistent=False,
        )

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, T, C)
            attention_mask: (B, T) with 1 = real token, 0 = pad. Optional.
        Returns:
            y: (B, T, C)
        """
        batch_size, seq_len, n_embd = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        att = q.matmul(k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.causal_mask[:seq_len, :seq_len], float("-inf"))
        if attention_mask is not None:
            # mask pad KEYS; large finite negative so pad QUERY rows stay NaN-free
            pad = attention_mask[:, None, None, :] == 0
            att = att.masked_fill(pad, torch.finfo(att.dtype).min)

        scores = F.softmax(att, dim=-1)
        out = scores.matmul(v)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        return self.c_proj(out)


# =====================================================================================
# MLP (provided)
# =====================================================================================


class MLP(nn.Module):
    """GPT-2 MLP: Linear up 4x, exact GELU (not the tanh approximation), Linear down."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x), approximate="none"))


# =====================================================================================
# Transformer block (provided)
# =====================================================================================


class Block(nn.Module):
    """Pre-LN residual block: x = x + attn(ln1(x)); x = x + mlp(ln2(x))."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = ManualLayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = ManualLayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


# =====================================================================================
# [FILL 1.2] — Full GPT-2
# =====================================================================================


class GPT(nn.Module):
    """
    GPT-2. Module names match HF so load_gpt2_from_hf is a rename + transpose:

        wte  : Embedding(vocab_size, n_embd)     token embeddings
        wpe  : Embedding(block_size, n_embd)     learned position embeddings
        drop : Dropout(p)                        0.0 for everything in this course
        h    : ModuleList of Block * n_layer
        ln_f : final LayerNorm

    The LM head is TIED to wte: logits = hidden @ wte.weight.T. Do not register a
    separate Linear — that would be 38M extra params and would break weight loading.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = ManualLayerNorm(config.n_embd)

        # GPT-2 init: normal(0, 0.02), residual projections scaled by 1/sqrt(2*n_layer)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_hidden(
        self,
        idx: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Final hidden states (B, T, C) — everything except the LM head. The reward
        model and the PPO value head consume these directly.

        Recipe:
            pos = positions_from_mask(idx, attention_mask)        # (B or 1, T)
            x   = drop(wte(idx) + wpe(pos))
            for block in h:  x = block(x, attention_mask)
            return ln_f(x)
        """
        B, T = idx.shape
        assert T <= self.config.block_size
        pos = positions_from_mask(idx, attention_mask)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.h:
            if self.config.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, attention_mask, use_reentrant=False
                )
            else:
                x = block(x, attention_mask)
        return self.ln_f(x)

    def forward(
        self,
        idx: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Logits (B, T, V) via the tied LM head."""
        hidden = self.forward_hidden(idx, attention_mask)
        return hidden @ self.wte.weight.t()

    # ---------------------------------------------------------------------------------
    # Sampling (provided)
    # ---------------------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Naive sampling loop, no KV cache. Returns (B, T0 + max_new_tokens); rows that
        emit EOS are padded with EOS afterwards.
        """
        device = idx.device
        T0 = idx.size(1)
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-8)
            logits = filter_logits(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
            if eos_token_id is not None:
                next_id = torch.where(
                    finished, torch.full_like(next_id, eos_token_id), next_id
                )
                finished = finished | (next_id == eos_token_id)
            idx = torch.cat([idx, next_id.unsqueeze(1)], dim=1)
            if eos_token_id is not None and bool(finished.all()):
                break
        # keep a fixed output length: pad early-finished batches with EOS
        short = T0 + max_new_tokens - idx.size(1)
        if short > 0:
            pad = torch.full((idx.size(0), short), eos_token_id, device=device, dtype=idx.dtype)
            idx = torch.cat([idx, pad], dim=1)
        return idx


def filter_logits(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Top-k and/or nucleus filtering on (B, V) logits. Filtered entries -> -inf."""
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, [-1]]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # keep the first token whose cumulative prob exceeds p
        cutoff = cum - torch.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(
            -1, sorted_idx, sorted_logits
        )
    return logits


# =====================================================================================
# Load HF weights (provided)
# =====================================================================================


def load_gpt2_from_hf(model: GPT, hf_name: str = "gpt2") -> GPT:
    """
    Copy HuggingFace GPT-2 weights into `model` in place.

    HF stores c_attn/c_proj/c_fc as Conv1D with weight shape (in, out); our nn.Linear
    stores (out, in), so those weights are transposed on the way in. `transformers`
    is used only here, only to download the checkpoint.
    """
    from transformers import GPT2LMHeadModel

    hf = GPT2LMHeadModel.from_pretrained(hf_name)
    sd_hf = hf.state_dict()

    transpose_suffixes = (
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    )

    sd = model.state_dict()
    loaded = set()
    for hf_key, tensor in sd_hf.items():
        if hf_key == "lm_head.weight":
            continue  # tied to wte
        if hf_key.endswith(("attn.bias", "attn.masked_bias")):
            continue  # HF's causal-mask buffers
        our_key = hf_key.removeprefix("transformer.")
        if our_key not in sd:
            raise KeyError(f"no parameter named {our_key} in our model (from {hf_key})")
        if our_key.endswith(transpose_suffixes):
            tensor = tensor.t()
        if sd[our_key].shape != tensor.shape:
            raise ValueError(
                f"shape mismatch for {our_key}: ours {tuple(sd[our_key].shape)} "
                f"vs HF {tuple(tensor.shape)}"
            )
        sd[our_key].copy_(tensor)
        loaded.add(our_key)

    missing = {k for k in sd if k not in loaded and "causal_mask" not in k}
    if missing:
        raise KeyError(f"parameters never loaded from HF: {sorted(missing)}")
    return model


# =====================================================================================
# Scalar head (provided) — reward head (Part 3) and value head (Part 5)
# =====================================================================================


class ScalarHead(nn.Module):
    """Linear(n_embd, 1): hidden states (B, T, C) -> per-token scalars (B, T)."""

    def __init__(self, n_embd: int):
        super().__init__()
        self.proj = nn.Linear(n_embd, 1)
        nn.init.normal_(self.proj.weight, std=1.0 / math.sqrt(n_embd + 1))
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden).squeeze(-1)
