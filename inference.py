"""Simple inference script for qualitative inspection of GPT checkpoints."""

from __future__ import annotations

import torch

from data import TokenizerBundle, build_tokenizer
from gpt import GPT, GPTConfig


def load_model(weights_path: str, device: torch.device) -> GPT:
    """Load a ``GPT`` instance from ``weights_path`` and move it to ``device``."""

    config = GPTConfig()
    model = GPT(config)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def prepare_prompt(
    tokenizer: TokenizerBundle, prompt: str, device: torch.device
) -> torch.Tensor:
    """Convert ``prompt`` into a tensor including the BOS token."""

    tokens = [tokenizer.bos] + tokenizer.encode(prompt)
    tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    return tensor


def generate_text(
    model: GPT,
    tokenizer: TokenizerBundle,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
) -> str:
    """Generate text from ``model`` given ``prompt`` and decode it to a string."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    device = next(model.parameters()).device
    input_tokens = prepare_prompt(tokenizer, prompt, device)
    sequence = model.generate(
        input_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        eos_token=tokenizer.eos,
    )
    generated_tokens = sequence[0].tolist()

    # Drop the leading BOS and stop at EOS if present.
    if tokenizer.bos in generated_tokens:
        generated_tokens = generated_tokens[1:]
    if tokenizer.eos in generated_tokens:
        eos_index = generated_tokens.index(tokenizer.eos)
        generated_tokens = generated_tokens[:eos_index]

    return tokenizer.encoder.decode(generated_tokens)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = "weights/sft.pt"
    prompt = "Human: Explain the importance of reinforcement learning from human feedback.\nAssistant:"

    tokenizer = build_tokenizer()
    model = load_model(weights_path, device)
    completion = generate_text(
        model, tokenizer, prompt, max_new_tokens=120, temperature=0.8
    )

    print("Prompt:\n" + prompt)
    print("\nModel completion:\n" + completion)


if __name__ == "__main__":
    main()
