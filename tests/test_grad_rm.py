"""
tests/test_grad_rm.py — Module 3.3

Gradient check + translation-invariance check for the Bradley–Terry loss.
"""

import torch

from grad_check import check_grad
from train_rm import bt_loss


def test_bt_loss_grad_wrt_chosen():
    torch.manual_seed(0)
    r_c = torch.randn(16, dtype=torch.float64, requires_grad=True)
    r_r = torch.randn(16, dtype=torch.float64)
    def loss_fn(x):
        return bt_loss(x, r_r)
    rel_err = check_grad(loss_fn, r_c, atol=1e-6)
    assert rel_err < 1e-6


def test_bt_loss_grad_wrt_rejected():
    torch.manual_seed(1)
    r_c = torch.randn(16, dtype=torch.float64)
    r_r = torch.randn(16, dtype=torch.float64, requires_grad=True)
    def loss_fn(x):
        return bt_loss(r_c, x)
    rel_err = check_grad(loss_fn, r_r, atol=1e-6)
    assert rel_err < 1e-6


def test_bt_loss_translation_invariant():
    """L(r_c, r_r) == L(r_c + C, r_r + C) for any scalar C."""
    torch.manual_seed(2)
    r_c = torch.randn(32, dtype=torch.float64)
    r_r = torch.randn(32, dtype=torch.float64)
    la = bt_loss(r_c, r_r).item()
    lb = bt_loss(r_c + 17.0, r_r + 17.0).item()
    assert abs(la - lb) < 1e-10


def test_bt_loss_symmetry_of_grads():
    """∂L/∂r_c + ∂L/∂r_r == 0 elementwise (exact negatives)."""
    torch.manual_seed(3)
    r_c = torch.randn(8, dtype=torch.float64, requires_grad=True)
    r_r = torch.randn(8, dtype=torch.float64, requires_grad=True)
    loss = bt_loss(r_c, r_r)
    loss.backward()
    assert torch.allclose(r_c.grad + r_r.grad, torch.zeros_like(r_c), atol=1e-12)