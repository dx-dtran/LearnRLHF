# LearnRLHF

This repository contains a compact reference implementation of supervised fine-tuning (SFT) and PPO-based reinforcement learning from human feedback (RLHF) for GPT-style language models.

## Hardware guidance

The table below estimates the memory required to train members of the GPT-2 family on a single NVIDIA RTX 4090 (24 GB). Parameter memory assumes mixed-precision training with AdamW, which typically stores weights, gradients, and two FP32 optimizer moments (~12 bytes per parameter). An additional 2–4 GB is usually needed for activations, temporary buffers, and dataloader queues.

| Model        | Parameters | AdamW state (≈12 B/param) | Estimated total VRAM need |
|--------------|------------|---------------------------|---------------------------|
| GPT-2        | 124 M      | ≈1.4 GB                  | <4 GB                     |
| GPT-2 Medium | 355 M      | ≈4.0 GB                  | ~6–8 GB                  |
| GPT-2 Large  | 774 M      | ≈8.7 GB                  | ~12–14 GB                |
| GPT-2 XL     | 1.558 B    | ≈17.4 GB                 | ~21–23 GB                |

With careful configuration (batch sizes of 1–2, sequence lengths ≤1024, activation checkpointing, and mixed precision), GPT-2 XL is typically the largest GPT-2 variant that fits within 24 GB while leaving a couple of gigabytes for activations and CUDA overhead. Larger effective batch sizes can be achieved with gradient accumulation.

## Supervised fine-tuning

`train_sft.py` now exposes a `train_sft` function and uses simple module-level constants instead of `argparse`.

```python
from train_sft import train_sft

train_sft(
    "data/sft_train.jsonl",
    out_dir="weights",
    batch_size=4,
    lr=5e-5,
    epochs=1,
    accum=1,
    warmup=100,
    init_path=None,
    dropout=0.0,
)
```

Running `python train_sft.py` will execute the same routine using the constants defined under the `if __name__ == "__main__"` guard—update them before launching training.

## PPO-based RLHF

`train_ppo.py` provides a `train_ppo` function configured in the same style. You can either import and call it directly:

```python
from train_ppo import train_ppo

train_ppo(
    "data/preferences_train.jsonl",
    sft_path="weights/sft_epoch_1.pt",
    out_dir="ppo_weights",
    dropout=0.0,
    rm_batch=4,
    rm_epochs=1,
    rm_lr=1e-5,
    ppo_batch=4,
    ppo_epochs=1,
    ppo_steps=4,
    max_new=64,
    ppo_lr=1e-5,
    clip=0.2,
    kl_coef=0.1,
    entropy_coef=0.01,
)
```

or edit the constants at the bottom of `train_ppo.py` and run `python train_ppo.py`.

Both training scripts now assume single-GPU execution. Gradient accumulation is available through the `accum` (SFT) and `ppo_steps` parameters.

## Testing

Unit tests (including gradient checks for both the SFT loss and the PPO policy objective) can be run with:

```bash
pytest
```
