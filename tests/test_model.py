"""
tests/test_model.py — Part 1.

Attention correctness (vs. an independent reference), causality, padding behavior,
full-GPT shapes, tied weights, sampling, and (slow/optional) HF weight-load parity.
"""

import pytest
import torch
import torch.nn.functional as F

from config import GPTConfig
from model import (
    GPT,
    Block,
    CausalSelfAttention,
    ManualLayerNorm,
    MLP,
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
# LayerNorm (provided code, kept honest)
# -------------------------------------------------------------------------------------


def test_manual_layernorm_matches_torch():
    torch.manual_seed(0)
    C = 16
    ln_ours = ManualLayerNorm(C).double()
    ln_ref = torch.nn.LayerNorm(C, eps=1e-5).double()
    ln_ref.weight.data.copy_(ln_ours.weight.data)
    ln_ref.bias.data.copy_(ln_ours.bias.data)

    x = torch.randn(4, 8, C, dtype=torch.float64, requires_grad=True)
    y_ours = ln_ours(x)
    y_ref = ln_ref(x)
    torch.testing.assert_close(y_ours, y_ref, atol=1e-10, rtol=0)

    g = torch.randn_like(y_ours)
    (y_ours * g).sum().backward()
    gw_ours = ln_ours.weight.grad.clone()
    gb_ours = ln_ours.bias.grad.clone()
    (y_ref * g).sum().backward()
    torch.testing.assert_close(gw_ours, ln_ref.weight.grad, rtol=1e-6, atol=1e-10)
    torch.testing.assert_close(gb_ours, ln_ref.bias.grad, rtol=1e-6, atol=1e-10)


# -------------------------------------------------------------------------------------
# [FILL 1.1] attention
# -------------------------------------------------------------------------------------


def test_attention_shapes():
    torch.manual_seed(0)
    attn = CausalSelfAttention(TINY)
    x = torch.randn(2, 10, TINY.n_embd)
    assert attn(x).shape == x.shape


def test_attention_is_causal():
    """Changing a future token must not change any earlier position's output."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(TINY).eval()
    x = torch.randn(1, 8, TINY.n_embd)
    y1 = attn(x).clone()
    x2 = x.clone()
    x2[0, -1, :] += 7.0
    y2 = attn(x2)
    torch.testing.assert_close(y1[0, :-1], y2[0, :-1], atol=1e-6, rtol=0)


def test_attention_matches_reference():
    """
    Independent reference: same c_attn/c_proj weights, but the attention math done
    by F.scaled_dot_product_attention. Catches missing scaling, missing softmax,
    and a forgotten output projection.
    """
    torch.manual_seed(0)
    attn = CausalSelfAttention(TINY).double().eval()
    B, T, C = 2, 12, TINY.n_embd
    nh, hs = TINY.n_head, C // TINY.n_head
    x = torch.randn(B, T, C, dtype=torch.float64)

    qkv = attn.c_attn(x)
    q, k, v = qkv.split(C, dim=-1)
    q, k, v = (t.view(B, T, nh, hs).transpose(1, 2) for t in (q, k, v))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = attn.c_proj(ref.transpose(1, 2).reshape(B, T, C))

    torch.testing.assert_close(attn(x), ref, atol=1e-10, rtol=1e-8)


def test_attention_padding_is_finite_and_ignored():
    """
    Left-padded rows must (a) produce finite outputs everywhere — a fully-masked pad
    query softmaxes to NaN if you mask with -inf — and (b) give real positions the
    same output as the unpadded sequence.
    """
    torch.manual_seed(0)
    attn = CausalSelfAttention(TINY).double().eval()
    T_real, T_pad = 6, 3
    x_real = torch.randn(1, T_real, TINY.n_embd, dtype=torch.float64)
    x_padded = torch.cat(
        [torch.randn(1, T_pad, TINY.n_embd, dtype=torch.float64), x_real], dim=1
    )
    mask = torch.cat([torch.zeros(1, T_pad), torch.ones(1, T_real)], dim=1)

    y_padded = attn(x_padded, attention_mask=mask)
    assert torch.isfinite(y_padded).all(), "pad rows must not go NaN/inf"

    y_real = attn(x_real)
    torch.testing.assert_close(y_padded[:, T_pad:], y_real, atol=1e-10, rtol=1e-8)


def test_mlp_and_block_shapes():
    torch.manual_seed(0)
    x = torch.randn(2, 10, TINY.n_embd)
    assert MLP(TINY)(x).shape == x.shape
    assert Block(TINY)(x).shape == x.shape


# -------------------------------------------------------------------------------------
# [FILL 1.2] full GPT
# -------------------------------------------------------------------------------------


def test_gpt_forward_shapes():
    torch.manual_seed(0)
    m = GPT(TINY)
    idx = torch.randint(0, TINY.vocab_size, (2, 12))
    assert m(idx).shape == (2, 12, TINY.vocab_size)
    assert m.forward_hidden(idx).shape == (2, 12, TINY.n_embd)


def test_gpt_tied_embeddings():
    """LM head must reuse wte — no separate lm_head Linear."""
    m = GPT(TINY)
    assert not hasattr(m, "lm_head"), "tie the LM head to wte, don't register a Linear"


def test_gpt_left_padding_consistent():
    """
    The course's padding contract, end to end: a left-padded row must produce the
    SAME last-token logits as the unpadded row. Requires both the key-padding mask
    and mask-derived position ids to be wired through.
    """
    torch.manual_seed(0)
    m = GPT(TINY).double().eval()
    ids = torch.randint(0, TINY.vocab_size, (1, 7))
    pad = torch.zeros(1, 4, dtype=torch.long)
    ids_padded = torch.cat([pad, ids], dim=1)
    mask = torch.cat([torch.zeros(1, 4), torch.ones(1, 7)], dim=1)

    logits_plain = m(ids)[:, -1, :]
    logits_padded = m(ids_padded, attention_mask=mask)[:, -1, :]
    torch.testing.assert_close(logits_padded, logits_plain, atol=1e-8, rtol=1e-8)


# -------------------------------------------------------------------------------------
# Sampling (provided)
# -------------------------------------------------------------------------------------


def test_generate_produces_correct_length():
    torch.manual_seed(0)
    m = GPT(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (2, 5))
    out = m.generate(idx, max_new_tokens=7, temperature=1.0)
    assert out.shape == (2, 12)


def test_generate_eos_padding():
    torch.manual_seed(0)
    m = GPT(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (2, 5))
    eos = 3
    out = m.generate(idx, max_new_tokens=20, eos_token_id=eos)
    assert out.shape == (2, 25)
    for b in range(2):
        gen = out[b, 5:].tolist()
        if eos in gen:
            after = gen[gen.index(eos):]
            assert all(t == eos for t in after), "tokens after EOS must be EOS padding"


# -------------------------------------------------------------------------------------
# Config sizes
# -------------------------------------------------------------------------------------


def test_config_from_name():
    cfg = GPTConfig.from_name("gpt2-medium")
    assert (cfg.n_layer, cfg.n_head, cfg.n_embd) == (24, 16, 1024)
    assert cfg.vocab_size == 50257
    with pytest.raises(ValueError):
        GPTConfig.from_name("gpt3")


# -------------------------------------------------------------------------------------
# HF weight-load parity (slow, needs network + transformers)
# -------------------------------------------------------------------------------------


@pytest.mark.slow
def test_hf_parity():
    pytest.importorskip("transformers")
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    from model import load_gpt2_from_hf

    model = GPT(GPTConfig()).eval()
    load_gpt2_from_hf(model, "gpt2")

    hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    ids = tok("Hello, world! This is a parity check.", return_tensors="pt").input_ids

    with torch.no_grad():
        ours = model(ids)
        theirs = hf(ids).logits
    max_diff = (ours - theirs).abs().max().item()
    assert max_diff < 1e-4, f"hf parity failed, max abs diff = {max_diff}"
