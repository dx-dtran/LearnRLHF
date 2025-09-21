import torch

from gpt import GPT, GPTConfig


def test_padded_sequences_produce_finite_logits():
    torch.manual_seed(0)
    config = GPTConfig(
        vocab_size=32,
        block_size=4,
        n_layer=1,
        n_head=2,
        n_embd=8,
        dropout=0.0,
    )
    model = GPT(config)
    tokens = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ],
        dtype=torch.long,
    )

    logits, _ = model(tokens, attention_mask=attention_mask)

    assert torch.isfinite(logits).all()
