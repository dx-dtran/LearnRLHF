import os
import sys

import torch
from torch.autograd import gradcheck

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from gpt import GPTConfig, GPT  # noqa: F401
from train_sft import supervised_loss  # noqa: E402


def test_supervised_loss_gradcheck():
    torch.manual_seed(0)
    batch = 2
    seq = 3
    vocab = 5

    logits = torch.randn(batch, seq, vocab, dtype=torch.double, requires_grad=True)
    targets = torch.randint(0, vocab, (batch, seq), dtype=torch.long)
    mask = torch.ones(batch, seq, dtype=torch.double)

    def objective(logits_input: torch.Tensor) -> torch.Tensor:
        return supervised_loss(logits_input, targets, mask)

    assert gradcheck(objective, (logits,), eps=1e-6, atol=1e-4, rtol=1e-4)


def test_supervised_loss_ignores_prompt_tokens():
    torch.manual_seed(0)
    batch = 1
    seq = 4
    vocab = 3

    logits = torch.zeros(batch, seq, vocab, dtype=torch.double, requires_grad=True)
    targets = torch.tensor([[0, 1, 2, 0]], dtype=torch.long)
    label_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.double)

    base_loss = supervised_loss(logits, targets, label_mask)

    # Changing prompt logits should not affect the loss as they are masked out.
    modified_logits = logits.clone().detach()
    modified_logits[0, 0, :] = torch.tensor([10.0, -10.0, 0.0], dtype=torch.double)
    modified_logits[0, 1, :] = torch.tensor([-5.0, 5.0, 0.0], dtype=torch.double)
    modified_logits.requires_grad_(True)
    masked_loss = supervised_loss(modified_logits, targets, label_mask)

    assert torch.allclose(base_loss, masked_loss)
