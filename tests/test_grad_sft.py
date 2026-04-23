"""
tests/test_grad_sft.py — Module 2.2

Gradient check for the masked CE loss + mask-respect test.
"""

import torch

from grad_check import check_grad
from train_sft import sft_loss


def _make(dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    B, T, V = 2, 5, 11
    logits = torch.randn(B, T, V, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, V, (B, T))
    # random 0/1 mask with at least one 1 per row
    loss_mask = (torch.rand(B, T, dtype=dtype) > 0.3).to(dtype)
    loss_mask[:, 0] = 1.0
    return logits, labels, loss_mask


def test_sft_loss_gradient():
    logits, labels, loss_mask = _make()
    def loss_fn(x):
        return sft_loss(x, labels, loss_mask)
    rel_err = check_grad(loss_fn, logits, atol=1e-5)
    assert rel_err < 1e-5


def test_sft_loss_respects_mask():
    """Changing a label at a masked-out position must not change the loss."""
    logits, labels, loss_mask = _make()
    # find a position with mask == 0
    zero_pos = (loss_mask == 0).nonzero()
    if zero_pos.numel() == 0:
        # rebuild with a forced zero
        loss_mask = loss_mask.clone()
        loss_mask[0, 0] = 0.0
        zero_pos = (loss_mask == 0).nonzero()
    b, t = zero_pos[0].tolist()
    loss_a = sft_loss(logits.detach(), labels, loss_mask).item()
    labels_b = labels.clone()
    labels_b[b, t] = (labels[b, t] + 1) % logits.size(-1)
    loss_b = sft_loss(logits.detach(), labels_b, loss_mask).item()
    assert abs(loss_a - loss_b) < 1e-12, \
        "loss must be invariant to labels at masked-out positions"


def test_sft_loss_scale_invariance_to_pad_only_rows():
    """
    A row of all-zero loss_mask should contribute nothing. (Only meaningful once you've
    added a clamp_min(1) guard in the denominator.)
    """
    logits, labels, loss_mask = _make()
    # force row 1 to be entirely masked out
    loss_mask = loss_mask.clone()
    loss_mask[1] = 0.0
    loss_all = sft_loss(logits.detach(), labels, loss_mask).item()
    loss_row0 = sft_loss(
        logits.detach()[:1], labels[:1], loss_mask[:1]
    ).item()
    assert abs(loss_all - loss_row0) < 1e-10