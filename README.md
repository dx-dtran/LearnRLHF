# LearnRLHF

This repository contains a compact reference implementation of supervised fine-tuning (SFT) and PPO-based reinforcement learning from human feedback (RLHF) for GPT-style language models.

## Hardware guidance

The RLHF workflow in this repository has three GPU-intensive stages: supervised fine-tuning (SFT) of the base policy, reward-model (RM) fitting, and PPO training of the aligned policy. The tables below summarize typical memory footprints on a single NVIDIA RTX 4090 (24 GB) when using mixed precision and AdamW (weights + gradients + 2 FP32 moments ≈12 bytes/parameter). These numbers assume sequence lengths ≤1024, batch sizes of 1–2 per optimizer step, fused kernels disabled, and 2–3 GB of headroom for activations, temporary buffers, and dataloader queues. Activation checkpointing or gradient accumulation can stretch the limits further if needed.

### Stage-by-stage VRAM usage

| Stage        | Components resident in memory | Approximate VRAM need |
|--------------|--------------------------------|-----------------------|
| SFT          | Policy model + optimizer state | `model_state + 3–4 GB` |
| RM training  | Reward model + optimizer state | `model_state + 3–4 GB` |
| PPO          | Policy + reference model, value head, optimizer state, rollout buffers | `2 × model_state + 5–6 GB` |

The “model_state” term corresponds to the parameter + optimizer memory for one GPT-2 checkpoint in the tables below.

### GPT-2 memory estimates

| Model        | Parameters | AdamW state (≈12 B/param) | SFT / RM VRAM | PPO VRAM (policy + ref) |
|--------------|------------|---------------------------|---------------|-------------------------|
| GPT-2        | 124 M      | ≈1.4 GB                  | ~4–5 GB       | ~8–9 GB                 |
| GPT-2 Medium | 355 M      | ≈4.0 GB                  | ~7–8 GB       | ~15–16 GB               |
| GPT-2 Large  | 774 M      | ≈8.7 GB                  | ~12–13 GB     | ~23–24 GB               |
| GPT-2 XL     | 1.558 B    | ≈17.4 GB                 | ~20–21 GB     | ~37–38 GB               |

Because PPO must hold both the current policy and a frozen reference copy in memory, GPT-2 Large is usually the practical ceiling on a single RTX 4090 without offloading or tensor parallelism. GPT-2 XL can be trained for SFT/RM but will exceed device memory once PPO requires two copies of the model. Techniques such as parameter-efficient adapters, ZeRO optimizer sharding, or CPU offload can push beyond these limits, but they are outside the scope of this compact reference implementation.

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
