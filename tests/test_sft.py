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
