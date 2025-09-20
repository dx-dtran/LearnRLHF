# LearnRLHF

Minimal PyTorch reference for supervised fine-tuning (SFT) and PPO-based RLHF on GPT-style models.

## Training scripts

Both entry points are written for single-GPU runs. Adjust the configuration objects at the bottom of each file before launching a job.

### Supervised fine-tuning

```bash
python train_sft.py
```

Key options (edit the `SFTConfig` instance):

- `data_path`: path to Anthropic HH SFT JSONL data.
- `out_dir`: directory for epoch checkpoints.
- `batch_size`, `lr`, `epochs`, `accum`, `warmup`, `dropout`: training hyperparameters.
- `init_path`: optional checkpoint to warm start from.

### PPO RLHF

```bash
python train_ppo.py
```

Tune the `PPOConfig` instance for RLHF runs:

- `preference_path`: Anthropic HH preference JSONL file.
- `sft_path`: optional SFT policy checkpoint used to seed the policy/reference/value/reward models.
- `rm_*`: reward model batch size, epochs, and learning rate.
- `ppo_*`: PPO batch sizing, number of epochs, steps per batch, rollout length, and optimizer rate.
- `clip`, `kl`, `entropy`: PPO objective weights.
- `out_dir`: directory that receives PPO checkpoints.

### Testing

Run the unit suite (includes gradient checks for SFT and PPO):

```bash
pytest
```

## GPT-2 sizing on a 24 GB RTX 4090

The table below assumes float32 training with Adam (≈16 bytes per parameter covering weights, gradients, and optimizer states) and a batch of one sequence with length 1024. Activation estimates cover forward and backward caches for the transformer blocks and leave headroom for temporary buffers. Mixed precision or gradient checkpointing can stretch the limits further.

| Model | Parameters | Param+optimizer memory | Estimated activations | Fits in 24 GB? | Notes |
|-------|------------|-------------------------|-----------------------|----------------|-------|
| GPT-2 Small | 124M | ≈1.9 GB | ≈0.1 GB | ✅ Comfortable | Leaves ample room for larger batches. |
| GPT-2 Medium | 355M | ≈5.3 GB | ≈0.2 GB | ✅ Comfortable | Practical for moderate batch sizes. |
| GPT-2 Large | 774M | ≈11.6 GB | ≈0.4 GB | ✅ Feasible | Expect ~12 GB remaining for activations, buffers, and dataloader overhead; gradient accumulation recommended. |
| GPT-2 XL | 1.5B | ≈23.2 GB | ≈0.6 GB | ❌ Not practical | Parameters plus optimizer states already exceed available VRAM before activations. |

In practice, GPT-2 Large (774M) is the largest variant that can be trained comfortably on a single RTX 4090 in float32. Dropping to bfloat16/float16 or using optimizer state sharding can shrink memory enough to experiment with slightly larger models, but 1.5B remains out of reach without offloading or model parallelism.
