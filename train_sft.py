import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from data import (
    TokenizerBundle,
    SupervisedDataset,
    build_tokenizer,
    collate_supervised,
)
from gpt import GPT, GPTConfig, load_gpt_checkpoint


def supervised_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross entropy between ``logits`` and ``targets`` masked on padding.

    The mask marks the valid positions inside each sequence.  ``targets`` is the
    input sequence shifted by one position which makes the language modelling
    objective explicit: the model learns to predict the next token ``y`` from the
    observed prefix ``x``.
    """

    vocab = logits.size(-1)
    flat_logits = logits.view(-1, vocab)
    flat_targets = targets.view(-1)
    flat_mask = mask.view(-1).to(logits.dtype)

    per_token = torch.nn.functional.cross_entropy(
        flat_logits,
        flat_targets,
        reduction="none",
        ignore_index=-1,
    )
    denom = flat_mask.sum().clamp(min=1.0)
    return (per_token * flat_mask).sum() / denom


def _resolve_initial_state(
    init_checkpoint: Optional[str], bundle: TokenizerBundle, block_size: int
) -> tuple[GPTConfig, dict | None]:
    config = GPTConfig(vocab_size=bundle.encoder.n_vocab, block_size=block_size)
    state = None
    if init_checkpoint:
        if not os.path.exists(init_checkpoint):
            raise FileNotFoundError(
                f"Initial checkpoint {init_checkpoint} does not exist; provide a Torch state dict"
            )
        config, state = load_gpt_checkpoint(init_checkpoint)
        if config.vocab_size != bundle.encoder.n_vocab:
            raise ValueError(
                "Tokenizer vocabulary does not match the checkpoint embedding size"
            )
        if block_size > config.block_size:
            raise ValueError(
                "Requested block_size exceeds the positional embeddings in the checkpoint"
            )
        config.block_size = min(block_size, config.block_size)
    return config, state


def train_sft(
    data_path: str,
    *,
    out_dir: str = "weights",
    batch_size: int = 8,
    lr: float = 5e-5,
    epochs: int = 3,
    grad_accumulation_steps: int = 1,
    init_path: Optional[str] = None,
    block_size: int = 1024,
    device: Optional[torch.device] = None,
) -> str:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Supervised data not found at {data_path}. Prepare the JSONL file before training."
        )

    bundle = build_tokenizer()
    config, state = _resolve_initial_state(init_path, bundle, block_size)
    model = GPT(config).to(device)

    if state is not None:
        model.load_state_dict(state, strict=False)

    dataset = SupervisedDataset(data_path, block_size=config.block_size, bundle=bundle)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_supervised(batch, bundle.pad),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    if grad_accumulation_steps < 1:
        raise ValueError("grad_accumulation_steps must be a positive integer")

    for epoch_idx in range(epochs):
        optimizer.zero_grad()
        step_in_epoch = 0
        running_loss = 0.0
        total_tokens = 0
        correct_tokens = 0
        for batch in loader:
            tokens = batch["input_ids"].to(device)
            targets = batch["target_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            logits, _ = model(tokens, attention_mask=mask)
            loss = supervised_loss(logits, targets, mask)

            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                valid = (targets != -1) & mask.bool()
                correct_tokens += (predictions.eq(targets) & valid).sum().item()
                total_tokens += valid.sum().item()
                running_loss += loss.item()

            (loss / grad_accumulation_steps).backward()
            step_in_epoch += 1

            if step_in_epoch % grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        remainder = step_in_epoch % grad_accumulation_steps
        if remainder != 0:
            scale = grad_accumulation_steps / remainder
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.mul_(scale)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        average_loss = running_loss / max(step_in_epoch, 1)
        accuracy = correct_tokens / max(total_tokens, 1)
        perplexity = torch.exp(torch.tensor(average_loss)).item()
        print(
            f"Epoch {epoch_idx + 1}/{epochs}: loss={average_loss:.4f}, "
            f"perplexity={perplexity:.2f}, token_accuracy={accuracy:.4f}"
        )

    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "sft.pt")
    torch.save(model.state_dict(), output_path)
    return output_path


def main() -> None:
    data_path = "data/hh_rlhf_sft_train.jsonl"
    out_dir = "weights"
    batch_size = 8
    lr = 5e-5
    epochs = 3
    init_path = "weights/gpt2.pt"
    grad_accumulation_steps = 1
    block_size = 1024
    device = None

    device_obj = torch.device(device) if device else None
    train_sft(
        data_path,
        out_dir=out_dir,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        grad_accumulation_steps=grad_accumulation_steps,
        init_path=init_path,
        block_size=block_size,
        device=device_obj,
    )


if __name__ == "__main__":
    main()
