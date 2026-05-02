# 05 — PPO training loop, memory, and stability

## Purpose

Theory and engineering notes for Module 5 (Problems 5.1 through 5.5).

By this point Module 4 has provided every individual function PPO needs:
sampling responses with their log-probs, computing per-token KL, shaping
rewards, computing advantages with GAE, and the policy / value / entropy
losses with their gradient checks. Module 5 ties those pieces into one
training script.

This note covers four practical things that the math notes do not:

1. The four GPT-2 networks that must be in memory simultaneously
   (trainable policy, value head sharing the policy backbone, frozen
   reference model, frozen reward model), and how to fit all of them on
   a single 24GB GPU.
2. The outer-loop / inner-loop structure of a PPO iteration: one
   rollout phase that runs without gradients, followed by K epochs of
   minibatch updates that recompute log-probs and values with gradients
   on.
3. What to log every iteration, and what trajectories of those logs
   indicate a healthy run versus reward hacking, mode collapse, or
   value-head divergence.
4. How to make the code work for all four GPT-2 sizes (small, medium,
   large, XL), even when the large variants run out of memory at full
   batch size.

Many PPO bugs do not show up as a failed gradient check; they show up
as wrong tensors flowing into otherwise correct loss functions. Common
causes are recomputing a quantity at the wrong time (`logprobs_old`
must be frozen at rollout time, not recomputed each minibatch),
forgetting `model.eval()` on the reference, or accidentally letting
gradients into the reward model. Treat this note as an execution
checklist as well as a theory note.

A one-line summary of one PPO iteration: sample responses from the
current policy on a batch of prompts, score them with the reward model,
measure how far the responses' per-token distribution drifted from the
frozen SFT reference, turn the scores plus the KL penalty into per-token
advantages, and take a few cautious optimizer steps that nudge the
policy toward higher-advantage tokens without letting it drift too far
from the rollout policy in a single update.

---

## 1. The four models

During PPO four GPT-2 networks are alive at the same time:

| Name      | Params | Trainable? | Gradients? | Optimizer state? | Role                                 |
|-----------|--------|------------|------------|------------------|--------------------------------------|
| Policy    | 124M   | yes        | yes        | yes (AdamW m, v) | Samples tokens; updated by PPO.      |
| Value     | shared backbone + 1 linear head | yes | yes | yes | Outputs `V(s_t)` per position.       |
| Reference | 124M   | **no**     | no         | no               | `pi_ref`; anchors the KL penalty.    |
| Reward    | 124M   | **no**     | no         | no               | Scores the response at the end.      |

The policy and value share a transformer backbone by default. The two
options:

- **Shared backbone** (the default): half the trainable memory. Value and
  policy gradients both flow through the backbone, which usually helps (the
  value loss provides a useful auxiliary task). Occasionally hurts if one of
  the two gradients dominates. Implementation: a single `GPT` module with
  two heads, `lm_head` (tied to `wte`, produces policy logits) and
  `value_head` (a `nn.Linear(n_embd, 1)`).
- **Separate networks**: more stable, but 2× the trainable memory and
  optimizer state. Some OpenAI recipes use this.

For 24GB the shared backbone is used. Scaling up to `gpt2-medium` and
finding instability is a one-line cue to try splitting off the value head.

A shared backbone means the value loss is not isolated. A large value
gradient updates the same hidden representations used by the policy logits.
The effect can be helpful (value learning teaches useful features), but the
value coefficient and clipping are not secondary details; they directly
affect the policy backbone.

### 1.1 Weight initialization

- **Policy**: load from `sft.pt`.
- **Reference**: load from `sft.pt` too (exact copy, then frozen).
  `ref_model.eval()` and set all params to `requires_grad_(False)`.
- **Reward model**: load from `rm.pt` (it already has its trained scalar
  head).
- **Value head**: zero-init the linear weights and bias. This makes
  `V_0(s) ≡ 0` at the start. Advantages equal returns initially, and the
  value head learns from scratch without corrupting early policy updates.
  This pattern is standard in PPO implementations.

Zero-initializing only the value head does not make the whole model
uninformative. The shared backbone still contains SFT representations. The
zero head says "before seeing PPO returns, predict no extra shaped reward",
which is a conservative starting baseline.

---

## 2. Memory budget (Problem 5.1)

bf16 mixed precision throughout. Fp32 master weights and AdamW state are
stored only for trainable parameters. Activations are gradient-checkpointed.

### 2.1 Rough count for gpt2-small

- Weights, bf16 (2 bytes per param), 4 models: `4 * 124M * 2B ≈ 0.99 GB`.
- Fp32 master copy of trainable weights (policy + value head; backbone
  shared): `~0.5 GB`.
- AdamW first and second moments, both fp32: `2 * 0.5 GB = 1.0 GB`.
- Activations with gradient checkpointing (batch 4, seq 512, 2 trainable
  models forward): `~3–6 GB` depending on checkpoint strategy.
- Rollout buffers (`logprobs_old`, `values_old`, `ref_logprobs`,
  `advantages`, `returns`, `response_mask`, etc.), each `(B_rollout,
  T_response)` floats: `64 * 256 * 4B * 6 ≈ 0.4 GB`.
- CUDA workspace and general overhead: `1–2 GB`.

**Total: roughly 7–10 GB**, comfortably below the 24GB ceiling.

If measured memory is much higher than this, look for accidental gradient
tracking through the reference or reward model, missing `torch.no_grad()`
in rollout, or storing full logit tensors in the rollout buffer. Rollouts
should store gathered log-probs, not vocabulary-sized distributions for
every token.

#### Worked example: a memory-budget walk-through

Plug actual numbers into the rough count, for `gpt2-small` with the
default PPO config (rollout batch 64, response len 256, n_embd 768).

    weights, bf16, 4 models = 4 * 124e6 * 2 bytes = 992 MB
    fp32 master copy of trainable params (124M policy + 769 value head)
                            ≈ 124e6 * 4 bytes ≈ 496 MB
    AdamW first moment, fp32 = 124e6 * 4 ≈ 496 MB
    AdamW second moment, fp32 = 124e6 * 4 ≈ 496 MB
    --- subtotal weights + optimizer state ≈ 2480 MB ≈ 2.5 GB

    rollout buffers, fp32:
      logprobs_old:  (64, 256)        =  16 384 floats ≈ 64 KB
      values_old:    (64, 256)        =  16 384 floats ≈ 64 KB
      ref_logprobs:  (64, 256)        =  16 384 floats ≈ 64 KB
      advantages:    (64, 256)        =  16 384 floats ≈ 64 KB
      returns:       (64, 256)        =  16 384 floats ≈ 64 KB
      response_mask: (64, 256)        =  16 384 floats ≈ 64 KB
      response_ids:  (64, 256), int64 =  16 384 longs  ≈ 128 KB
    --- rollout buffers ≈ 0.5 MB total

    activations under gradient checkpointing:
      one trainable forward+backward at batch 4 micro, seq 512 ≈ 3-6 GB
      depending on how many layer outputs you cache vs. recompute

    CUDA workspace + cuBLAS scratch ≈ 1-2 GB

    grand total ≈ 7-10 GB

Two observations:

- The rollout buffers are tiny next to the optimizer state. Storing a full
  `(B, T, V)` logit tensor instead of a gathered `(B, T)` log-prob tensor
  would add `64 * 256 * 50257 * 4 bytes ≈ 3.3 GB` per buffer. The shape
  hint above prevents this bug.
- Activations dominate the variable part of the budget. If memory is
  tight, the cheapest knob is shorter `response_len` or smaller per-step
  batch; both cut activation memory linearly, while the optimizer-state
  line is fixed.

### 2.2 gpt2-medium (355M) is tight

Weights and optimizer state both scale roughly linearly, so medium is about
3× larger. To fit:

- Smaller `rollout_batch_size` (try 16 instead of 64).
- Possibly smaller `response_max_len` (128 instead of 256).
- Gradient checkpointing definitely on.

### 2.3 Memory check script (Problem 5.1)

Write a small script that instantiates all four models, runs one dummy
forward, and prints `torch.cuda.max_memory_allocated()`. Log the output
into this note for every size that fits.

Reset peak memory stats before the dummy forward so the number reflects the
operation being measured. Record the batch size, sequence length, dtype,
and whether gradient checkpointing is enabled. A memory number without
those conditions is hard to interpret later.

---

## 3. The outer loop (Problem 5.2)

One PPO iteration has two phases: a **rollout** phase (no gradients,
generate data) and an **optimize** phase (K epochs of gradient steps on
that data). Pseudocode:

```
for iter in range(num_iters):

    # --- ROLLOUT PHASE (no grad on policy) ---
    prompts = sample_prompt_batch()
    with torch.no_grad():
        response_ids, logprobs_old, values_old, mask = \
            generate_with_logprobs(policy, value_head, prompts)
        ref_logprobs = gather_logprobs(ref_model(prompts + response_ids), response_ids)
        rm_rewards   = reward_model(prompts + response_ids)  # one scalar per row
        kl_t         = logprobs_old - ref_logprobs
        per_tok_r    = shape_reward(rm_rewards, kl_t, mask, kl_coef)
        advantages, returns = gae(per_tok_r, values_old, mask, gamma, lam)
        advantages = normalize_advantages(advantages, mask)

    rollout = {
        "prompts": prompts,
        "response_ids": response_ids,
        "logprobs_old": logprobs_old,
        "values_old": values_old,
        "advantages": advantages,
        "returns": returns,
        "mask": mask,
    }

    # --- OPTIMIZE PHASE (grads on policy and value backbone) ---
    for epoch in range(K):
        for mb in minibatches(rollout):
            logits_new, values_new = policy_and_value(mb.prompts + mb.response_ids)
            logprobs_new = gather_logprobs(logits_new, mb.response_ids)
            L_pi = ppo_policy_loss(logprobs_new, mb.logprobs_old, mb.advantages, mb.mask, eps)
            L_v  = value_loss(values_new, mb.values_old, mb.returns, mb.mask, eps_v)
            H    = masked_entropy(logits_new, mb.mask)
            loss = L_pi + c_v * L_v - c_ent * H
            opt.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()

    log_stats(iter, ...)
```

Three invariants:

- **`logprobs_old` is frozen at rollout time** and never recomputed during
  the inner epochs. It is the reference point for the importance ratio
  `rho_t`.
- **`logprobs_new` is recomputed fresh in every minibatch**, since that is
  where the policy gradient flows.
- **All advantages and returns are stop-gradient.** Detach everything
  coming out of the rollout before storing it.

A fourth practical invariant: frozen models stay frozen. The reference and
reward model should be in eval mode, under `torch.no_grad()`, and have
`requires_grad_(False)`. Any optimizer parameter group that accidentally
includes them is both a memory bug and an algorithm bug.

#### Worked example: tensor shapes for one PPO iteration

Concrete sizes for the default config (`B = 4` for the inner-loop
micro-batch, prompt_len ≤ 64, response_len = 32, n_embd = 768, vocab =
50257). The rollout phase produces tensors at the *rollout* batch size
(64); the inner loop slices them into the micro-batch size (4).

```
rollout phase, no grad:
  prompts (left-padded):   (64,  64)            int64
  response_ids:            (64,  32)            int64
  attention_mask:          (64,  96)            bool        # prompt + response
  response_mask:           (64,  32)            float32     # 1 on real response tokens
  logprobs_old:            (64,  32)            float32     # gathered, not (B, T, V)
  values_old:              (64,  32)            float32
  ref_logprobs:            (64,  32)            float32
  rm_rewards:              (64,)                float32     # one scalar per row
  per_tok_r:               (64,  32)            float32
  advantages:              (64,  32)            float32     # normalized
  returns:                 (64,  32)            float32

inner-loop micro-batch (4 rows):
  mb.prompts:              (4,   64)            int64
  mb.response_ids:         (4,   32)            int64
  mb.logprobs_old:         (4,   32)            float32
  mb.values_old:           (4,   32)            float32
  mb.advantages:           (4,   32)            float32
  mb.returns:              (4,   32)            float32
  mb.mask:                 (4,   32)            float32

  forward pass:
    logits_new:            (4,   96, 50257)     bf16        # full sequence
    logprobs_new:          (4,   32)            bf16        # gathered at response positions only
    values_new:            (4,   96)            bf16        # then sliced to (4, 32)
```

Two checks worth doing in code:

- `logprobs_old` and `logprobs_new` have the same shape `(mb_B,
  response_len)`, not `(mb_B, prompt_len + response_len)`. Aligning prompt
  log-probs with response log-probs is the most common off-by-one source.
- `logits_new` has the full vocab dimension `(B, T, V)` *only* during the
  forward pass; gather to `(B, T)` immediately and discard the big tensor
  before backprop, otherwise activations balloon.

---

## 4. The inner loop (Problem 5.3)

### 4.1 Minibatch sampling

Shuffle a permutation of rollout indices at the start of each epoch, then
iterate in chunks of `minibatch_size`. Each minibatch covers all batch
rows and all time steps; the effective batch dim for the inner loop is
essentially `(rollout_batch_size * T_response)` flattened.

A simpler alternative is to slice by row, keeping whole sequences
together. Either works; row-wise is slightly easier to get the masking
right.

For a teaching implementation, row-wise minibatches are usually worth the
small loss of randomness. Keeping complete response sequences together
makes it easier to inspect one row's prompt, response, old log-probs,
advantages, and mask when something looks wrong.

### 4.2 K epochs

`K = 4` is the standard choice. More epochs amortize rollout cost better
(less generation per step of compute), but also let the policy drift
further from `pi_old`. The clip handles some of that drift, but not
indefinitely. A clip fraction above 40% indicates `K` should be lowered.

PPO has both an outer iteration count and inner epochs. The rollout policy
is fixed for one outer iteration. Every inner epoch makes the current
policy less like that rollout policy. At some point the data is too
stale, and clipping turns most tokens into zero-gradient cases.

### 4.3 Gradient flow

- **Policy gradient** enters via `logprobs_new` inside `L_policy`.
- **Value gradient** enters via `values_new` inside `L_value`.
- **Entropy gradient** enters via `logits_new` inside `H`.
- All three backpropagate through the shared backbone in one backward
  pass. The policy and the value head are not stepped separately.

Stepping separately would make the second loss see parameters already
changed by the first loss, which complicates the meaning of the PPO
update. Combine the terms, run one backward pass, clip one combined
gradient norm, take one optimizer step.

---

## 5. Logging (Problem 5.4)

Log at least the following per outer iteration. A CSV with one row per
iteration is enough; emit matplotlib plots every 50 iterations to inspect
trajectories.

| Metric             | Expected trajectory                                                      |
|--------------------|--------------------------------------------------------------------------|
| `mean_rm_reward`   | Trends up slowly.                                                        |
| `mean_kl_k3`       | Starts near 0, drifts up to a small positive value (~1–6 nats per response). If it explodes, raise `beta` or lower the policy lr. |
| `L_pi`             | Bounces around zero. Should not blow up or pin at zero.                  |
| `L_v`              | Decreases over time as the value head learns.                            |
| `entropy`          | Drifts down slowly from the SFT-level entropy.                           |
| `clip_fraction`    | Rises from 0 into the 10–30% range. Above 40% means updates are too aggressive; lower `K` or lr. |
| `grad_norm`        | Pre-clip may spike. Post-clip, sits around O(1). Stable is healthy. |
| `tokens_per_sec`   | Roughly constant. Drops signal memory pressure or disk / IO issues.         |
| `response_length`  | Watch for a slow creep upward; that is length bias in the RM showing up. |

Save sample generations as part of the logging routine. Every few
iterations, save a small fixed set of prompts and responses. Plots show
what the optimizer is doing; samples show what the model is becoming.

### 5.1 Healthy pattern

Reward climbs, KL climbs slowly, entropy drifts down slowly, clip fraction
sits around 20%, grad norm is stable, tokens / sec is stable.

### 5.2 Unhealthy patterns and fixes

- **Reward up, KL also up fast, responses look weird.** Reward hacking.
  Raise `beta` or lower the policy lr. Also inspect the RM: is reward
  strongly correlated with response length? A length-bias problem in the
  RM may be the cause.
- **Reward flat, KL flat.** Policy is not moving. Lower `beta`, raise lr.
- **Clip fraction above 50%.** Updates too large for PPO's clip to absorb.
  Lower `K`, lower lr, or smaller minibatches.
- **Entropy crashes toward 0 in the first 50 iterations.** Premature
  determinism. Set `c_entropy = 0.01`.
- **Grad norm exploding.** Lower lr. Inspect the value loss in particular,
  which is usually the value head overshooting. Confirm that the value
  clip `eps_v` is on and working.

When a run fails, change one knob at a time. PPO metrics are coupled:
lowering LR can reduce KL, clip fraction, and reward growth all at once.
Changing LR, beta, K, and response length together makes the cause of
any improvement impossible to identify.

### 5.3 Worked example: healthy vs. reward-hacking log rows

Two CSV rows, both real-shaped, both from the same training script.

**Healthy iteration (iter = 80):**

```
iter,  mean_rm_reward,  mean_kl_k3,  L_pi,    L_v,    entropy,  clip_fraction,  grad_norm,  tok/s,  resp_len
80,    +0.42,           0.61,        -0.018,  0.143,  2.71,     0.18,           0.62,       1180,   91.4
```

Reward is climbing slowly. KL is small but positive (about 0.6 nats per
response, well within the soft trust region for `beta = 0.02`). Policy
loss hovers near zero, value loss is decreasing run over run, entropy is
drifting down from the SFT baseline of around 3.0, clip fraction is in
the healthy 10–30% band, gradient norm is well under the clip threshold
of 1.0, and tokens / sec is steady. Average response length has risen
modestly.

**Reward hacking (iter = 80, different run):**

```
iter,  mean_rm_reward,  mean_kl_k3,  L_pi,    L_v,    entropy,  clip_fraction,  grad_norm,  tok/s,  resp_len
80,    +1.85,           4.20,        -0.105,  0.880,  1.42,     0.51,           1.00,       1170,   232.7
```

Reward has shot up four-fold. KL has crossed 4 nats per response and is
trending up faster than reward. Entropy collapsed from ~2.7 to ~1.4. Clip
fraction is over 50%, meaning most tokens land outside `[1 - eps, 1 +
eps]` per inner step. Gradient norm is pinned at the clip ceiling (1.0
exactly), suggesting most batches saturate the clip. Average response
length exploded from ~90 to ~233. This is the classic length-bias plus
reward-hack pattern: the policy is producing long, high-reward,
low-entropy answers that look nothing like SFT.

Fixes, in order: raise `beta` (0.02 → 0.06), then lower the policy
learning rate by 3×, then check the RM for a length-vs-reward correlation
on a held-out batch. Make one change per run.

---

## 6. Config scaling (Problem 5.5)

Add a classmethod to `GPTConfig`:

```python
@classmethod
def from_name(cls, name: str) -> "GPTConfig":
    return {
        "gpt2-small":  cls(n_layer=12, n_head=12, n_embd=768),
        "gpt2-medium": cls(n_layer=24, n_head=16, n_embd=1024),
        "gpt2-large":  cls(n_layer=36, n_head=20, n_embd=1280),
        "gpt2-xl":     cls(n_layer=48, n_head=25, n_embd=1600),
    }[name]
```

Smoke test: for each size, instantiate the model, run one forward +
backward at `batch=1, seq=64`, and record
`torch.cuda.max_memory_allocated()`. `gpt2-small` and `gpt2-medium`
should step cleanly on a 24GB card. `gpt2-large` and `gpt2-xl` may OOM,
which is acceptable; the requirement is only that the code compiles and
the forward shapes are correct, not that it trains at full scale.

Log the memory table into this note.

The scaling exercise also tests whether dimensions are centralized.
Changing model size by editing constants in multiple files indicates the
config abstraction is not doing its job, which makes the rest of the
code fragile.

---

## 7. What to commit to `notes/05-ppo.md`

After Module 5, add:

- Memory printout table for each GPT-2 size that fits.
- One training run's CSV (head and tail rows at least), or plots of
  reward / KL / entropy / clip fraction over iterations.
- One paragraph on any instability seen and how it was fixed.
- A sanity-check comparison: greedy generations from `rlhf.pt` versus
  `sft.pt` on 3 held-out prompts. RLHF should be recognizably different,
  ideally better.
