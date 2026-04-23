"""
model.py — GPT-2 from scratch.

Modules this file covers:
    1.1  Config + embeddings
    1.2  LayerNorm
    1.3  Causal self-attention
    1.4  MLP (GELU)
    1.5  Transformer block
    1.6  Full GPT-2
    1.7  Load HF weights
    1.8  Sampling

Style: nanoGPT. Flat, few abstractions. Write small, write clear, don't optimize.

We deliberately do NOT use `nn.LayerNorm` in Module 1.2 — you implement it. After you've
passed the gradient check there, you may switch to `torch.nn.LayerNorm` everywhere else
for perf (but keep your hand-rolled version around for reference).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


# =====================================================================================
# Problem 1.2 — LayerNorm by hand
# =====================================================================================

class ManualLayerNorm(nn.Module):
    """
    Layer normalization you implement from scratch. No nn.LayerNorm under the hood.

    Forward (last dim = C):
        mu  = mean(x, dim=-1, keepdim=True)
        var = mean((x - mu)^2, dim=-1, keepdim=True)       # BIASED variance (divide by C)
        xhat = (x - mu) / sqrt(var + eps)
        y    = gamma * xhat + beta

    Notes:
        - GPT-2's LN uses biased variance (divide by n, not n-1). Autograd's
          `torch.nn.LayerNorm` does too. Don't use .var(unbiased=False).mean(); just
          compute it explicitly.
        - gamma / beta shape = (n_embd,), affine always on for GPT-2.
        - eps = 1e-5 to match HF.

    TODO(1.2): implement __init__ and forward. Your unit test will compare against
    torch.nn.LayerNorm on random fp64 inputs: max abs diff < 1e-10 forward,
    < 1e-6 relative on parameter gradients.
    """

    def __init__(self, n_embd: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # TODO(1.2): create self.weight (init 1s) and self.bias (init 0s) of shape (n_embd,)
        raise NotImplementedError("TODO(1.2): ManualLayerNorm.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(1.2): compute LN by hand. No calls to F.layer_norm or torch.nn.LayerNorm.
        raise NotImplementedError("TODO(1.2): ManualLayerNorm.forward")


# =====================================================================================
# Problem 1.3 — Causal self-attention
# =====================================================================================

class CausalSelfAttention(nn.Module):
    """
    Single multi-head causal self-attention block.

    Parameters (match GPT-2 naming so weight loading is a transpose away):
        c_attn : Linear(n_embd, 3 * n_embd, bias=True)    # fused Q,K,V
        c_proj : Linear(n_embd, n_embd, bias=True)        # output projection

    Forward:
        qkv = c_attn(x)                                   # (B, T, 3C)
        q,k,v = split on last dim, reshape to (B, nh, T, hs)
        att = (q @ k^T) / sqrt(hs)                        # (B, nh, T, T)
        apply causal mask (upper triangle = -inf)
        if attention_mask is not None (key padding, 1=keep, 0=pad):
            att.masked_fill_( ~mask[:, None, None, :].bool(), -inf )
        p = softmax(att, dim=-1)
        out = p @ v                                       # (B, nh, T, hs)
        reshape -> (B, T, C); c_proj

    Hints:
        - For Module 1.3, implement it with the explicit matmul / masked_fill / softmax
          chain (that's what you're learning). Only after your grad check passes may you
          swap in torch.nn.functional.scaled_dot_product_attention for speed.
        - Never forget contiguous() after a transpose before view().
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size

        # TODO(1.3): create self.c_attn and self.c_proj
        # Both use `bias=config.bias` (True for GPT-2).

        # Register a causal mask buffer you can slice with [:T, :T].
        # TODO(1.3): self.register_buffer("causal_mask", torch.tril(torch.ones(T, T)).bool())

        raise NotImplementedError("TODO(1.3): CausalSelfAttention.__init__")

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:              (B, T, C)
            attention_mask: (B, T) with 1=real token, 0=pad. Optional.

        Returns:
            y: (B, T, C)
        """
        # TODO(1.3): implement.
        raise NotImplementedError("TODO(1.3): CausalSelfAttention.forward")


# =====================================================================================
# Problem 1.4 — MLP (GELU 4x)
# =====================================================================================

class MLP(nn.Module):
    """
    GPT-2 MLP:
        c_fc   : Linear(n_embd, 4*n_embd)
        gelu   : exact GELU (NOT tanh approx)
        c_proj : Linear(4*n_embd, n_embd)

    TODO(1.4): implement. Use F.gelu(x, approximate='none').
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        # TODO(1.4)
        raise NotImplementedError("TODO(1.4): MLP.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(1.4)
        raise NotImplementedError("TODO(1.4): MLP.forward")


# =====================================================================================
# Problem 1.5 — Transformer block (pre-LN residual)
# =====================================================================================

class Block(nn.Module):
    """
    Pre-LN transformer block (GPT-2 style, NOT post-LN like the original Transformer):

        x = x + attn(ln1(x))
        x = x + mlp(ln2(x))

    TODO(1.5): implement. Use your ManualLayerNorm from 1.2 (or switch to nn.LayerNorm
    after 1.2's grad check passes — document which one you chose).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        # TODO(1.5): self.ln_1, self.attn, self.ln_2, self.mlp
        raise NotImplementedError("TODO(1.5): Block.__init__")

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # TODO(1.5)
        raise NotImplementedError("TODO(1.5): Block.forward")


# =====================================================================================
# Problem 1.6 — Full GPT-2
# =====================================================================================

class GPT(nn.Module):
    """
    Modules (names match HF so load_gpt2_from_hf is a rename + transpose):

        wte  : Embedding(vocab_size, n_embd)
        wpe  : Embedding(block_size, n_embd)
        drop : Dropout(p)                                # usually 0 for us
        h    : ModuleList of Block * n_layer
        ln_f : final LayerNorm

    Tied weights: the LM head reuses wte.weight transposed; do NOT add a separate Linear.

    Forward (`attention_mask` optional, (B,T) 1/0 for key-padding):
        B, T = idx.shape
        pos  = arange(T)
        x    = drop(wte(idx) + wpe(pos))
        for block in h: x = block(x, attention_mask)
        x = ln_f(x)
        logits = x @ wte.weight.T                        # (B, T, V)

    In Module 4 (PPO) you'll also want a `forward_hidden(idx)` that returns the final
    hidden states (before the LM head) so the value head and RM head can consume them.
    Add that as part of Problem 1.6.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # TODO(1.6): register self.wte, self.wpe, self.drop, self.h, self.ln_f
        raise NotImplementedError("TODO(1.6): GPT.__init__")

    def forward(
        self,
        idx: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits of shape (B, T, V)."""
        # TODO(1.6)
        raise NotImplementedError("TODO(1.6): GPT.forward")

    def forward_hidden(
        self,
        idx: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return final hidden states of shape (B, T, C). Needed by RM + value head."""
        # TODO(1.6)
        raise NotImplementedError("TODO(1.6): GPT.forward_hidden")

    # -----------------------------------------------------------------------
    # Problem 1.8 — sampling
    # -----------------------------------------------------------------------

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
        Args:
            idx:             (B, T0) prompt ids
            max_new_tokens:  int
            temperature:     float > 0
            top_k:           int, 0 disables
            top_p:           float in (0,1], 1.0 disables
            eos_token_id:    if given, stop generating for a row once it emits EOS.

        Returns:
            idx: (B, T0 + T_generated). If `eos_token_id` is given, rows that hit EOS are
            padded with `eos_token_id` after their EOS.

        TODO(1.8): implement a simple naive loop (no KV cache for now).
            - crop context to last block_size tokens each step
            - logits[:, -1, :] / temperature
            - optional top-k: keep only top_k logits, rest = -inf
            - optional top-p (nucleus): sort descending, cumsum softmax, mask tokens whose
              cumulative probability is above p (keep the first token that exceeds p)
            - sample with torch.multinomial on softmax
            - append and continue

        Do NOT return log-probs from this function. Sampling-with-logprobs for PPO lives
        in ppo_core.generate_with_logprobs — a separate function, because PPO needs the
        logprobs tensor attached to the computational graph (kind of — see 4.1).
        """
        # TODO(1.8)
        raise NotImplementedError("TODO(1.8): GPT.generate")


# =====================================================================================
# Problem 1.7 — Load HF weights
# =====================================================================================

def load_gpt2_from_hf(model: GPT, hf_name: str = "gpt2") -> GPT:
    """
    Load HuggingFace GPT-2 weights into our `model` in place.

    HF uses a custom `Conv1D` layer (which stores weight with shape (in, out)) whereas we
    use `nn.Linear` (which stores (out, in)). So for these four parameter-name suffixes
    you need to TRANSPOSE the HF weight before copying in:

        attn.c_attn.weight
        attn.c_proj.weight
        mlp.c_fc.weight
        mlp.c_proj.weight

    Recommended approach:
        from transformers import GPT2LMHeadModel
        hf = GPT2LMHeadModel.from_pretrained(hf_name)
        sd_hf = hf.state_dict()

    Build a key map from HF names -> our names:
        transformer.wte.weight          -> wte.weight
        transformer.wpe.weight          -> wpe.weight
        transformer.ln_f.{weight,bias}  -> ln_f.{weight,bias}
        transformer.h.{i}.ln_1.{w,b}    -> h.{i}.ln_1.{w,b}
        transformer.h.{i}.attn.c_attn.{w,b} -> h.{i}.attn.c_attn.{w,b}   (transpose w)
        transformer.h.{i}.attn.c_proj.{w,b} -> h.{i}.attn.c_proj.{w,b}   (transpose w)
        transformer.h.{i}.ln_2.{w,b}    -> h.{i}.ln_2.{w,b}
        transformer.h.{i}.mlp.c_fc.{w,b}   -> h.{i}.mlp.c_fc.{w,b}       (transpose w)
        transformer.h.{i}.mlp.c_proj.{w,b} -> h.{i}.mlp.c_proj.{w,b}     (transpose w)
        lm_head.weight                  -> tied to wte.weight (skip; already tied)
        transformer.h.{i}.attn.bias / masked_bias -> skip (it's the causal mask buffer)

    After loading, copy into your model with `param.data.copy_(tensor)`.

    TODO(1.7): implement. Your test (tests/test_model.py::test_hf_parity) will assert
    that your model's logits match HF's on a fixed prompt within abs tol 1e-4.

    Note: we allow `transformers` ONLY for this one weight-load step. If you don't want
    the dependency, you can instead read `gpt2.safetensors` directly with `safetensors`,
    or run once in a scratch env to dump a plain state_dict to a .pt file and keep that.
    """
    # TODO(1.7)
    raise NotImplementedError("TODO(1.7): load_gpt2_from_hf")


# =====================================================================================
# Problem 3.1 — Reward model head / value head (tiny wrappers)
# =====================================================================================

class ScalarHead(nn.Module):
    """
    Linear(n_embd, 1) used as either the reward head (Module 3) or the value head
    (Module 5). Separate instance per purpose — do not share weights between RM and V.

    TODO(3.1 / 5.1): implement a trivial __init__ and forward.
    Forward takes hidden states (B, T, C) and returns (B, T) scores.
    """

    def __init__(self, n_embd: int):
        super().__init__()
        # TODO(3.1)
        raise NotImplementedError("TODO(3.1): ScalarHead.__init__")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # TODO(3.1)
        raise NotImplementedError("TODO(3.1): ScalarHead.forward")