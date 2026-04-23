"""
tests/test_model.py — Module 1.

Tiny-config forward/shape tests + LayerNorm parity + (optional) HF weight load parity.

The HF-parity test is marked slow/optional: it downloads ~500MB the first time it runs
and needs `transformers` installed. Run with `pytest -m slow tests/test_model.py`.
"""

import os

import pytest
import torch

from config import GPTConfig
from model import (
    Block,
    CausalSelfAttention,
    GPT,
    MLP,
    ManualLayerNorm,
)


TINY = GPTConfig(
    block_size=32,
    vocab_size=64,
    n_layer=2,
    n_head=2,
    n_embd=16,
    dropout=0.0,
    bias=True,
)


# -------------------------------------------------------------------------------------
# Problem 1.2
# -------------------------------------------------------------------------------------

def test_manual_layernorm_matches_torch():
    torch.manual_seed(0)
    C = 16
    ln_ours = ManualLayerNorm(C).double()
    ln_ref = torch.nn.LayerNorm(C, eps=1e-5).double()
    # copy params so they match
    ln_ref.weight.data.copy_(ln_ours.weight.data)
    ln_ref.bias.data.copy_(ln_ours.bias.data)

    x = torch.randn(4, 8, C, dtype=torch.float64, requires_grad=True)
    y_ours = ln_ours(x)
    y_ref = ln_ref(x)
    assert torch.allclose(y_ours, y_ref, atol=1e-10), \
        f"max diff = {(y_ours - y_ref).abs().max().item()}"

    # backward parity on parameter grads
    g = torch.randn_like(y_ours)
    (y_ours * g).sum().backward()
    gw_ours = ln_ours.weight.grad.clone()
    gb_ours = ln_ours.bias.grad.clone()

    ln_ours.weight.grad = None
    ln_ours.bias.grad = None
    (y_ref * g).sum().backward()
    assert torch.allclose(gw_ours, ln_ref.weight.grad, rtol=1e-6, atol=1e-10)
    assert torch.allclose(gb_ours, ln_ref.bias.grad, rtol=1e-6, atol=1e-10)


# -------------------------------------------------------------------------------------
# Problem 1.3 / 1.4 / 1.5 — shape smoke tests (quick)
# -------------------------------------------------------------------------------------

def test_attention_shapes():
    torch.manual_seed(0)
    attn = CausalSelfAttention(TINY)
    x = torch.randn(2, 10, TINY.n_embd)
    y = attn(x)
    assert y.shape == x.shape


def test_attention_is_causal():
    """
    Changing a future token must NOT change any earlier position's output.
    This is the single most common GPT-2 bug, so test it explicitly.
    """
    torch.manual_seed(0)
    attn = CausalSelfAttention(TINY).eval()
    x = torch.randn(1, 8, TINY.n_embd)
    y1 = attn(x).clone()
    x2 = x.clone()
    x2[0, -1, :] += 7.0  # perturb last position
    y2 = attn(x2)
    assert torch.allclose(y1[0, :-1], y2[0, :-1], atol=1e-6), \
        "changing token at position t should not affect positions < t (causal)"


def test_mlp_shapes():
    torch.manual_seed(0)
    m = MLP(TINY)
    x = torch.randn(2, 10, TINY.n_embd)
    assert m(x).shape == x.shape


def test_block_shapes():
    torch.manual_seed(0)
    b = Block(TINY)
    x = torch.randn(2, 10, TINY.n_embd)
    assert b(x).shape == x.shape


# -------------------------------------------------------------------------------------
# Problem 1.6 — full GPT
# -------------------------------------------------------------------------------------

def test_gpt_forward_shapes():
    torch.manual_seed(0)
    m = GPT(TINY)
    idx = torch.randint(0, TINY.vocab_size, (2, 12))
    logits = m(idx)
    assert logits.shape == (2, 12, TINY.vocab_size)


def test_gpt_hidden_shapes():
    torch.manual_seed(0)
    m = GPT(TINY)
    idx = torch.randint(0, TINY.vocab_size, (2, 12))
    h = m.forward_hidden(idx)
    assert h.shape == (2, 12, TINY.n_embd)


def test_gpt_tied_embeddings():
    """LM head must reuse wte — total params should NOT include a second vocab×embd."""
    m = GPT(TINY)
    n_params = sum(p.numel() for p in m.parameters())
    # Upper bound if UNtied: 2 * V * C + ...; tied: V*C + ...
    # Just assert param id: no `lm_head` linear registered
    assert not hasattr(m, "lm_head"), "tie wte with logits; don't register a separate lm_head"


# -------------------------------------------------------------------------------------
# Problem 1.7 — HF parity (slow, optional)
# -------------------------------------------------------------------------------------

@pytest.mark.slow
def test_hf_parity():
    transformers = pytest.importorskip("transformers")
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    cfg = GPTConfig()  # gpt2-small defaults
    model = GPT(cfg).eval()

    from model import load_gpt2_from_hf
    load_gpt2_from_hf(model, "gpt2")

    hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    ids = tok("Hello, world! This is a parity check.", return_tensors="pt").input_ids

    with torch.no_grad():
        ours = model(ids)
        theirs = hf(ids).logits
    max_diff = (ours - theirs).abs().max().item()
    assert max_diff < 1e-4, f"hf parity failed, max abs diff = {max_diff}"


# -------------------------------------------------------------------------------------
# Problem 1.8 — sampling
# -------------------------------------------------------------------------------------

def test_generate_produces_correct_length():
    torch.manual_seed(0)
    m = GPT(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (2, 5))
    out = m.generate(idx, max_new_tokens=7, temperature=1.0)
    assert out.shape == (2, 12)