import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from gpt import GPTConfig, GPT


def test_sft_gradient_matches_finite_difference():
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=16, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = GPT(config)
    model.eval()

    inputs = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    targets = torch.tensor([[2, 3, 4, -1]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)

    param = model.wte.weight
    index = (1, 0)

    model.zero_grad()
    _, loss = model(inputs, targets=targets, attention_mask=mask)
    loss.backward()
    autograd_grad = param.grad[index].item()

    eps = 1e-3
    with torch.no_grad():
        original = param[index].item()
        param[index] = original + eps
        plus_loss = model(inputs, targets=targets, attention_mask=mask)[1].item()
        param[index] = original - eps
        minus_loss = model(inputs, targets=targets, attention_mask=mask)[1].item()
        param[index] = original

    finite = (plus_loss - minus_loss) / (2 * eps)
    assert torch.isclose(torch.tensor(autograd_grad), torch.tensor(finite), atol=1e-3, rtol=1e-3)
