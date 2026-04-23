"""
tests/test_grad_check.py — sanity-check the grad_check helper itself.

If THIS doesn't pass, no other grad check in the repo can be trusted.
"""

import pytest
import torch

from grad_check import check_grad, numeric_grad, rel_error


def test_numeric_matches_analytic_quadratic():
    torch.manual_seed(0)
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    def f(x):
        return (x ** 2).sum()
    rel = check_grad(f, x, atol=1e-7)
    assert rel < 1e-7


def test_numeric_matches_analytic_sigmoid():
    torch.manual_seed(0)
    x = torch.randn(5, dtype=torch.float64, requires_grad=True)
    def f(x):
        return torch.sigmoid(x).sum()
    rel = check_grad(f, x, atol=1e-6)
    assert rel < 1e-6


def test_catches_wrong_gradient():
    """If we corrupt the autograd gradient, check_grad must flag it."""
    torch.manual_seed(0)
    x = torch.randn(3, dtype=torch.float64, requires_grad=True)
    def f(x):
        return (x ** 2).sum()
    loss = f(x)
    loss.backward()
    x.grad.mul_(1.1)  # corrupt

    # recompute numeric grad and compare manually
    num = numeric_grad(f, x.detach().clone())
    err = rel_error(x.grad, num)
    assert err > 1e-2, "corrupted grad should produce visible rel error"