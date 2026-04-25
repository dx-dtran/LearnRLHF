# 05 — PPO training loop, memory, and stability

## Purpose

Theory and engineering notes for Module 5 (Problems 5.1 through 5.5). The math is
all in `04-*.md`. This note zooms out and talks about:

1. The four models you need in memory simultaneously, and how to fit them on a
   24GB GPU.
2. The outer/inner loop structure of a PPO iteration.
3. What to log and what a healthy training run should look like.
4. How to make the code work for all four GPT-2 sizes, even if the larger ones
   OOM.

Module 5 is where correct small functions become a training system. The individual losses can
all pass gradient checks and the full PPO run can still fail because tensors are stale,
models are in the wrong mode, or logging hides the first sign of instability. Read this note
as an execution checklist as much as a theory note.

---

## 1. The four models

During PPO you have *four* GPT-2s alive at the same time:

| Name      | Params | Trainable? | Gradients? | Optimizer state? | Role                                 |
|-----------|--------|------------|------------|------------------|--------------------------------------|
| Policy    | 124M   | yes        | yes        | yes (AdamW m, v) | Samples tokens; updated by PPO.      |
| Value     | shared backbone + 1 linear head | yes | yes | yes | Outputs `V(s_t)` per position.       |
| Reference | 124M   | **no**     | no         | no               | `pi_ref` — anchors the KL penalty.   |
| Reward    | 124M   | **no**     | no         | no               | Scores the response at the end.      |

The policy and value **share a transformer backbone** by default. This is a design
choice and there's a real alternative:

- **Shared backbone** (our default): half the trainable memory. Value and policy
  gradients both flow through the backbone, which usually helps (the value loss
  provides a useful auxiliary task). Occasionally hurts if one of the two
  gradients dominates. Implementation: a single `GPT` module with two heads —
  `lm_head` (tied to `wte`, produces policy logits) and `value_head` (a
  `nn.Linear(n_embd, 1)`).
- **Separate networks**: more stable, but 2x the trainable memory and optimizer
  state. Some OpenAI recipes use this.

For 24GB we use shared. If you scale up to `gpt2-medium` and training feels
unstable, splitting off the value head is a one-line experiment worth trying.

Shared backbone means the value loss is not isolated. A large value gradient updates the same
hidden representations used by the policy logits. That can be helpful because value learning
teaches useful features, but it also means the value coefficient and clipping are not
secondary details. They directly affect the policy backbone.

### 1.1 Weight initialization

- **Policy**: load from `sft.pt`.
- **Reference**: load from `sft.pt` too (exact copy, then frozen).
  `ref_model.eval()` and set all params to `requires_grad_(False)`.
- **Reward model**: load from `rm.pt` (it already has its trained scalar head).
- **Value head**: zero-init the linear weights and bias. This makes
  `V_0(s) ≡ 0` at the start. Advantages equal returns initially, and the value
  head gets to learn fresh without corrupting early policy updates. Common
  folklore in PPO implementations.

Zero-initializing only the value head does not make the whole model uninformative. The shared
backbone still contains SFT representations. The zero head simply says "before seeing PPO
returns, predict no extra shaped reward." That is a conservative starting baseline.

---

## 2. Memory budget (Problem 5.1)

We use bf16 mixed precision throughout. Fp32 master weights and AdamW state are
stored only for trainable parameters. Activations are gradient-checkpointed.

### 2.1 Rough count for gpt2-small

- Weights, bf16 (2 bytes per param), 4 models: `4 * 124M * 2B ≈ 0.99 GB`.
- Fp32 master copy of trainable weights (policy + value head; backbone shared):
  `~0.5 GB`.
- AdamW first and second moments, both fp32: `2 * 0.5 GB = 1.0 GB`.
- Activations with gradient checkpointing (batch 4, seq 512, 2 trainable models
  forward): `~3–6 GB` depending on checkpoint strategy.
- Rollout buffers (`logprobs_old`, `values_old`, `ref_logprobs`, `advantages`,
  `returns`, `response_mask`, etc.), each `(B_rollout, T_response)` floats:
  `64 * 256 * 4B * 6 ≈ 0.4 GB`.
- CUDA workspace and general overhead: `1–2 GB`.

**Total: roughly 7–10 GB.** Plenty of headroom on a 24GB card.

If your measured memory is much higher than this, look for accidental gradient tracking
through the reference or reward model, missing `torch.no_grad()` in rollout, or storing full
logit tensors in the rollout buffer. Rollouts should store gathered log-probs, not
vocabulary-sized distributions for every token.

### 2.2 gpt2-medium (355M) is tight

Weights and optimizer state both scale roughly linearly, so medium is about 3x
larger. You'll need:

- Smaller `rollout_batch_size` (try 16 instead of 64).
- Possibly smaller `response_max_len` (128 instead of 256).
- Gradient checkpointing definitely on.

### 2.3 Memory check script (Problem 5.1)

Write a small script that instantiates all four models, runs one dummy forward,
and prints `torch.cuda.max_memory_allocated()`. Log the output into this note
for every size that fits.

Reset peak memory stats before the dummy forward so the number reflects the operation you are
measuring. Also record the batch size, sequence length, dtype, and whether gradient
checkpointing is enabled. A memory number without those conditions is hard to interpret
later.

---

## 3. The outer loop (Problem 5.2)

One PPO iteration has two phases: a **rollout** phase (no gradients, generate
data) and an **optimize** phase (K epochs of gradient steps on that data).
Pseudocode:

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

Three invariants to keep straight:

- **`logprobs_old` is *frozen* at rollout time** and never recomputed during the
  inner epochs. It's the reference point for the importance ratio `rho_t`.
- **`logprobs_new` is *recomputed fresh* in every minibatch** — that's where the
  policy gradient actually flows through.
- **All advantages and returns are stop-gradient.** Detach everything coming out
  of the rollout before storing it.

A fourth practical invariant: frozen models stay frozen. The reference and reward model
should be in eval mode, under `torch.no_grad()`, and have `requires_grad_(False)`. Any
optimizer parameter group that accidentally includes them is both a memory bug and an
algorithm bug.

---

## 4. The inner loop (Problem 5.3)

### 4.1 Minibatch sampling

Shuffle a permutation of rollout indices at the start of each epoch, then iterate
in chunks of `minibatch_size`. Each minibatch covers all batch rows and all time
steps — the "effective batch dim" for the inner loop is essentially
`(rollout_batch_size * T_response)` flattened.

A simpler alternative is to slice by row (whole sequences stay together). Either
works; row-wise is slightly easier to get the masking right.

For a teaching implementation, row-wise minibatches are usually worth the small loss of
randomness. Keeping complete response sequences together makes it easier to inspect one row's
prompt, response, old log-probs, advantages, and mask when something looks wrong.

### 4.2 K epochs

`K = 4` is the standard choice. More epochs amortize rollout cost better (less
generation per step of compute), but also let the policy drift further from
`pi_old`. The clip handles some of that drift, but not indefinitely. If you see
clip fraction above 40%, lower `K`.

This is why PPO has both an outer iteration count and inner epochs. The rollout policy is
fixed for one outer iteration. Every inner epoch makes the current policy less like that
rollout policy. At some point the data is too stale, and clipping turns most tokens into
zero-gradient cases.

### 4.3 Gradient flow

- **Policy gradient** enters via `logprobs_new` inside `L_policy`.
- **Value gradient** enters via `values_new` inside `L_value`.
- **Entropy gradient** enters via `logits_new` inside `H`.
- All three backpropagate through the shared backbone in one backward pass. You
  do **not** step the policy and the value head separately.

Stepping separately would make the second loss see parameters already changed by the first
loss, which complicates the meaning of the PPO update. Combine the terms, run one backward
pass, clip one combined gradient norm, and take one optimizer step.

---

## 5. Logging (Problem 5.4)

Log at least these per outer iteration. A CSV with one row per iteration is
enough; emit matplotlib plots every 50 iterations so you can eyeball trajectories.

| Metric             | Expected trajectory                                                      |
|--------------------|--------------------------------------------------------------------------|
| `mean_rm_reward`   | Trends up slowly.                                                        |
| `mean_kl_k3`       | Starts near 0, drifts up to a small positive value (~1–6 nats per response). If it explodes, raise `beta` or lower the policy lr. |
| `L_pi`             | Bounces around zero. Should not blow up or pin at zero.                  |
| `L_v`              | Decreases over time as the value head learns.                            |
| `entropy`          | Drifts down slowly from the SFT-level entropy.                           |
| `clip_fraction`    | Rises from 0 into the 10–30% range. Above 40% means updates are too aggressive — lower `K` or lr. |
| `grad_norm`        | Pre-clip may spike. Post-clip, should sit around O(1). Stable is healthy. |
| `tokens_per_sec`   | Roughly constant. Drops signal memory pressure or disk/IO issues.         |
| `response_length`  | Watch for a slow creep upward — that's length bias in the RM showing up. |

Add sample generations to the logging habit. Every few iterations, save a small fixed set of
prompts and responses. Plots tell you what the optimizer is doing; samples tell you what the
model is becoming.

### 5.1 Healthy pattern

Reward climbs, KL climbs slowly, entropy drifts down slowly, clip fraction sits
around 20%, grad norm stable, tokens/sec stable.

### 5.2 Unhealthy patterns and fixes

- **Reward up, KL also up fast, responses look weird.** Reward hacking. Raise
  `beta` or lower the policy lr. Also look at the RM — is reward strongly
  correlated with response length? Might be a length-bias problem in the RM.
- **Reward flat, KL flat.** Policy isn't moving at all. Lower `beta`, raise lr.
- **Clip fraction above 50%.** Updates too big for PPO's clip to absorb. Lower
  `K`, lower lr, or smaller minibatches.
- **Entropy crashes toward 0 in the first 50 iterations.** Premature determinism.
  Set `c_entropy = 0.01`.
- **Grad norm exploding.** Lower lr. Check the value loss in particular — usually
  it's the value head overshooting. Make sure the value clip `eps_v` is on and
  working correctly.

When a run fails, change one knob at a time. PPO metrics are coupled: lowering LR can reduce
KL, clip fraction, and reward growth all at once. If you change LR, beta, K, and response
length together, you will not know which change helped.

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

Smoke test: for each size, instantiate the model, run one forward + backward at
`batch=1, seq=64`, record `torch.cuda.max_memory_allocated()`. `gpt2-small` and
`gpt2-medium` should step cleanly on a 24GB card. `gpt2-large` and `gpt2-xl` may
OOM, which is fine — the requirement is just that the code compiles and the
forward shapes are correct, not that it trains at full scale.

Log the memory table into this note.

The scaling exercise is also a test of whether dimensions are centralized. If changing model
size requires editing constants in multiple files, the config abstraction is not doing its
job. The model may still run, but it will be fragile.

---

## 7. What to commit to `notes/05-ppo.md`

After Module 5, add:

- Memory printout table for each GPT-2 size that fits.
- One training run's CSV (head and tail rows at least), or plots of
  reward / KL / entropy / clip fraction over iterations.
- One paragraph on any instability you saw and how you fixed it.
- A sanity-check comparison: greedy generations from `rlhf.pt` versus `sft.pt`
  on 3 held-out prompts. RLHF should be recognizably different, ideally better.
