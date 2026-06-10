"""
config.py — every knob in one place.

One dataclass per phase. No nested configs, no yaml. Change model size with
GPTConfig.from_name("gpt2-medium") — everything else stays the same.
"""

from dataclasses import dataclass


# -------------------------------------------------------------------------------------
# Model config
# -------------------------------------------------------------------------------------

@dataclass
class GPTConfig:
    """GPT-2 architectural config. Defaults = gpt2-small (124M)."""

    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # GPT-2 uses bias everywhere
    gradient_checkpointing: bool = False  # trade compute for memory in training

    @classmethod
    def from_name(cls, name: str, **overrides) -> "GPTConfig":
        sizes = {
            "gpt2-small":  dict(n_layer=12, n_head=12, n_embd=768),    # 124M
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),   # 355M
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),   # 774M
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),   # 1558M
        }
        if name not in sizes:
            raise ValueError(f"unknown model name {name!r}; pick from {list(sizes)}")
        return cls(**{**sizes[name], **overrides})


# -------------------------------------------------------------------------------------
# Training configs: one dataclass per phase, no inheritance.
# -------------------------------------------------------------------------------------

@dataclass
class SFTConfig:
    # data
    block_size: int = 1024
    batch_size: int = 8             # per-step; effective batch = batch_size * accum_steps
    accum_steps: int = 8
    # optim
    lr: float = 3e-5
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 200
    min_lr_ratio: float = 0.1
    # run
    epochs: int = 2
    log_every: int = 10
    eval_every: int = 250
    eval_batches: int = 50
    save_path: str = "sft.pt"


@dataclass
class RMConfig:
    block_size: int = 1024
    batch_size: int = 4             # pairs per step (each pair is 2 forward passes)
    accum_steps: int = 8
    lr: float = 1e-5
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 100
    epochs: int = 1
    log_every: int = 10
    eval_every: int = 200
    eval_batches: int = 50
    init_from: str = "sft.pt"       # RM is initialized from the SFT model
    save_path: str = "rm.pt"


@dataclass
class PPOConfig:
    # rollout
    prompt_max_len: int = 512
    response_max_len: int = 128
    rollout_batch_size: int = 32    # prompts per rollout iteration
    temperature: float = 1.0        # keep 1.0: recorded logprobs assume raw softmax
    top_k: int = 0                  # 0 = disabled; keep off for PPO rollouts
    top_p: float = 1.0
    # PPO
    ppo_epochs: int = 4
    minibatch_size: int = 8
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    gamma: float = 1.0
    gae_lambda: float = 0.95
    # coefficients
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    kl_coef: float = 0.02           # beta in r_t = r_RM * 1{terminal} - beta * kl_t
    # optim
    policy_lr: float = 1e-6
    value_lr: float = 1e-5
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    # run
    num_iters: int = 500            # outer iterations (rollout + optimize)
    log_every: int = 1
    plot_every: int = 50
    save_every: int = 50
    log_csv: str = "ppo_log.csv"
    # checkpoints to load from
    policy_init: str = "sft.pt"
    ref_init: str = "sft.pt"
    rm_init: str = "rm.pt"
    save_path: str = "rlhf.pt"
