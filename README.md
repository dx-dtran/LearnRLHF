# LearnRLHF

This repository contains a compact reference implementation of supervised fine-tuning (SFT) and PPO-based reinforcement learning from human feedback (RLHF) for GPT-style language models.

## Hardware guidance

The RLHF workflow in this repository has three GPU-intensive stages: supervised fine-tuning (SFT) of the base policy, reward-model (RM) fitting, and PPO training of the aligned policy. The tables below summarize typical memory footprints on a single NVIDIA RTX 4090 (24 GB) when using mixed precision and AdamW (weights + gradients + 2 FP32 moments ≈12 bytes/parameter). These numbers assume sequence lengths ≤1024, batch sizes of 1–2 per optimizer step, fused kernels disabled, and 2–3 GB of headroom for activations, temporary buffers, and dataloader queues. Activation checkpointing or gradient accumulation can stretch the limits further if needed.

### Stage-by-stage VRAM usage

| Stage        | Components resident in memory | Approximate VRAM need |
|--------------|--------------------------------|-----------------------|
| SFT          | Policy model + optimizer state | `model_state + 3–4 GB` |
| RM training  | Reward model + optimizer state | `model_state + 3–4 GB` |
| PPO          | Policy + reference model with shared value head, reward scorer, optimizer state, rollout buffers | `2 × model_state + 3–4 GB` |

The “model_state” term corresponds to the parameter + optimizer memory for one GPT-2 checkpoint in the tables below.

### GPT-2 memory estimates

| Model        | Parameters | AdamW state (≈12 B/param) | SFT / RM VRAM | PPO VRAM (policy + ref) |
|--------------|------------|---------------------------|---------------|-------------------------|
| GPT-2        | 124 M      | ≈1.4 GB                  | ~4–5 GB       | ~7–8 GB                 |
| GPT-2 Medium | 355 M      | ≈4.0 GB                  | ~7–8 GB       | ~13–14 GB               |
| GPT-2 Large  | 774 M      | ≈8.7 GB                  | ~12–13 GB     | ~21–22 GB               |
| GPT-2 XL     | 1.558 B    | ≈17.4 GB                 | ~20–21 GB     | ~33–34 GB               |

Because PPO must hold both the current policy and a frozen reference copy in memory, GPT-2 Large remains the practical ceiling on a single RTX 4090 without offloading or tensor parallelism. With the shared transformer/value head introduced in this refactor we have verified that PPO fine-tuning of GPT-2 Large runs within 24 GB while leaving a small amount of headroom. GPT-2 XL can be trained for SFT/RM but will exceed device memory once PPO requires two copies of the model. Techniques such as parameter-efficient adapters, ZeRO optimizer sharding, or CPU offload can push beyond these limits, but they are outside the scope of this compact reference implementation.

The table below estimates the memory required to train members of the GPT-2 family on a single NVIDIA RTX 4090 (24 GB). Parameter memory assumes mixed-precision training with AdamW, which typically stores weights, gradients, and two FP32 optimizer moments (~12 bytes per parameter). An additional 2–4 GB is usually needed for activations, temporary buffers, and dataloader queues.

| Model        | Parameters | AdamW state (≈12 B/param) | Estimated total VRAM need |
|--------------|------------|---------------------------|---------------------------|
| GPT-2        | 124 M      | ≈1.4 GB                  | <4 GB                     |
| GPT-2 Medium | 355 M      | ≈4.0 GB                  | ~6–8 GB                  |
| GPT-2 Large  | 774 M      | ≈8.7 GB                  | ~12–14 GB                |
| GPT-2 XL     | 1.558 B    | ≈17.4 GB                 | ~21–23 GB                |

With careful configuration (batch sizes of 1–2, sequence lengths ≤1024, activation checkpointing, and mixed precision), GPT-2 XL is typically the largest GPT-2 variant that fits within 24 GB while leaving a couple of gigabytes for activations and CUDA overhead. Larger effective batch sizes can be achieved with gradient accumulation.

## Preparing the Anthropic HH dataset

Both the supervised and reward-model trainers expect JSONL files derived from the
[`Anthropic/hh-rlhf`](https://huggingface.co/datasets/Anthropic/hh-rlhf)
dataset. Each record supplied to the trainers follows one of the two minimal
schemas below:

```json
// Supervised fine-tuning (SFT)
{"prompt": "<conversation history>", "chosen": "<assistant reply>"}

// Reward modelling (RM)
{"prompt": "<conversation history>", "chosen": "<preferred reply>", "rejected": "<dispreferred reply>"}
```

You can export these files with the `datasets` library:

```bash
pip install datasets  # if you do not already have it

python - <<'PY'
from datasets import load_dataset

dataset = load_dataset("Anthropic/hh-rlhf", split="train")

# SFT: only the preferred continuations are required
sft = dataset.remove_columns([c for c in dataset.column_names if c not in {"prompt", "chosen"}])
sft.to_json("data/hh_rlhf_sft_train.jsonl")

# RM: keep both chosen and rejected responses
rm = dataset.remove_columns([c for c in dataset.column_names if c not in {"prompt", "chosen", "rejected"}])
rm.to_json("data/hh_rlhf_preferences_train.jsonl")
PY
```

Both trainers default to the file names shown above. You can adjust the paths by
editing the configuration variables inside `train_sft.py` and `train_rm.py`.

## Supervised fine-tuning

Edit the configuration block at the bottom of `train_sft.py` to point to your
SFT JSONL file, tweak the batch size or learning rate, and then launch training
with:

```bash
python train_sft.py
```

The defaults mirror a minimal run on the Anthropic HH SFT split. Set the
`device` variable to "cuda" or "cpu" if you need to force device selection, and
increase `grad_accumulation_steps` to enable gradient accumulation when memory
is limited.

## Reward model fitting

Similarly, edit the configuration block at the end of `train_rm.py` to select
your preference JSONL file, output path, and other hyperparameters, then run:

```bash
python train_rm.py
```

The reward-model script shares the same configuration style, including support
for reusing an SFT checkpoint via `init_path`, toggling gradient accumulation
through `grad_accumulation_steps`, and switching devices by setting `device`.

## PPO-based RLHF

`train_ppo.py` provides a `train_ppo` function configured in the same style. You
can either import and call it directly or edit the constants at the bottom of
`train_ppo.py` and run `python train_ppo.py`.

Both training scripts now assume single-GPU execution. Gradient accumulation is
available through the shared `grad_accumulation_steps` argument.

## Testing

Unit tests (including gradient checks for both the SFT loss and the PPO policy objective) can be run with:

```bash
pytest
```
