"""
grad_check.py — shared gradient-check helpers.

Every loss in this repo gets a gradient check using centered differences in fp64.
`check_grad` below is the only helper you should need. Use it like:

    def loss_fn(x): return (x ** 3).sum()
    x = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    check_grad(loss_fn, x)      # asserts rel_err < 1e-5

Key conventions:
    - Everything in fp64. Do NOT run grad checks in fp32 — you'll chase noise.
    - Tiny tensors. n_embd=16, heads=2, seq=8, batch=2.
    - Tolerance 1e-5 relative error is a reasonable default; tighten to 1e-7 for purely
      linear or quadratic losses, loosen to 1e-4 for anything with a hard clip.
"""

from typing import Callable

import torch


def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a - b).abs().max().item()
    den = (a.abs().max() + b.abs().max() + 1e-12).item()
    return num / den


def numeric_grad(
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Centered-difference gradient of scalar `loss_fn(x)` wrt x."""
    assert x.dtype == torch.float64, "grad checks must run in fp64"
    g = torch.zeros_like(x)
    flat = x.reshape(-1)
    gflat = g.reshape(-1)
    for i in range(flat.numel()):
        orig = flat[i].item()
        flat[i] = orig + eps
        lp = loss_fn(x).item()
        flat[i] = orig - eps
        lm = loss_fn(x).item()
        flat[i] = orig
        gflat[i] = (lp - lm) / (2 * eps)
    return g


def check_grad(
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float = 1e-6,
    atol: float = 1e-5,
) -> float:
    """
    Compare autograd grad of loss_fn w.r.t. x to centered-difference at fp64.
    Returns the relative error; raises AssertionError if it exceeds `atol`.
    """
    assert x.requires_grad and x.dtype == torch.float64
    if x.grad is not None:
        x.grad.zero_()
    loss = loss_fn(x)
    loss.backward()
    analytic = x.grad.detach().clone()

    x_nograd = x.detach().clone().requires_grad_(False)
    def _loss_no_grad(xx):
        # recreate the op without building grad
        return loss_fn(xx)
    numeric = numeric_grad(_loss_no_grad, x_nograd, eps=eps)

    err = rel_error(analytic, numeric)
    assert err < atol, f"grad check failed: rel_err={err:.3e} > atol={atol:.1e}"
    return err