# 05 — PPO training loop, memory, and stability

## Purpose

Theory and engineering notes for Module 5 (Problems 5.1 through 5.5). The math was in
the `04-*.md` files. Here we zoom out and talk about:

1. The four models you need in memory and how to fit them on a 24GB GPU.
2. The outer/inner loop structure of a PPO iteration.
3. What signals to log and what their healthy trajectories look like.
4. How to scale the model-size knob so the same codebase runs for
   `gpt2-small|medium|large|xl` even if the larger ones OOM.

---

## 1. The four models

During PPO you have *four* GPT-2s instantiated simultaneously:

| Name      | Params | Trainable? | Gradients? | Optimizer state? | Role                                 |
|-----------|--------|------------|------------|------------------|--------------------------------------|
| Policy    | 124M   | yes        | yes        | yes (AdamW m, v) | Samples actions; updated by PPO.     |
| Value     | shared backbone + 1 linear head | yes | yes | yes | Outputs $V(s_t)$ per position.       |
| Reference | 124M   | **no**     | no         | no               | $\pi_\text{ref}$ — KL penalty anchor.|
| Reward    | 124M   | **no**     | no         | no               | Terminal reward at end of response.  |

The policy and the value model **share the transformer backbone**. This is a design
choice (documented in Problem 5.1). Alternative: separate networks. Pros and cons:

- **Shared backbone** (our default): half the memory for the trainable models. Value
  and policy gradients interact in the backbone, which usually helps (the value
  signal provides an auxiliary task); can occasionally hurt if one head's gradient
  dominates. Implement by having a single `GPT` module with two heads: `lm_head`
  (for policy logits, tied to `wte`) and `value_head` (a `nn.Linear(n_embd, 1)`).
- **Separate networks**: more stable, 2x the trainable memory. Used in some OpenAI
  recipes.

For 24GB we use **shared**. If you scale up to `gpt2-medium` and training is unstable,
splitting the value head off is a one-line experiment.

### 1.1 Weight initialization

- Policy: from `sft.pt`.
- Reference: also from `sft.pt` (exact copy, frozen). `reference_model.eval()`, all
  params `.requires_grad_(False)`.
- Reward model: from `rm.pt` (has the scalar head already trained).
- Value head: **init the linear head weights to zero**, bias to zero. This makes
  $V_0 \equiv 0$ at the start — advantages equal returns, and the value head starts
  fresh without interfering with early policy updates. Common PPO folklore.

---

## 2. Memory budget (Problem 5.1)

Bf16 mixed precision throughout. Fp32 master weights + AdamW state on *trainable*
params only. Activations checkpointed.

### 2.1 Rough count for gpt2-small

- Weights (bf16, 2 bytes/param): $4 \times 124\text{M} \times 2\text{B} = 0.99$ GB
- Fp32 master copy of trainable (policy + value backbone, but they share): ~0.5 GB
- AdamW state (m, v in fp32): 2 $\times$ 0.5 GB = 1.0 GB
- Activations with gradient checkpointing (batch 4 × seq 512 × 2 trainable models
  forward): ~3–6 GB depending on checkpoint strategy
- Rollout buffers: `logprobs_old`, `values_old`, `ref_logprobs`, `advantages`,
  `returns`, `response_mask` — all $(B_\text{rollout}, T_\text{response})$ floats.
  $64 \times 256 \times 4\text{B} \times 6 \approx 0.4$ GB.
- CUDA workspace + overhead: ~1–2 GB.

**Total**: ~7–10 GB. Plenty of headroom on a 24GB card.

### 2.2 gpt2-medium (355M) is tight

Weights scale linearly (roughly 3x), as does optimizer state. You'll need:

- Smaller `rollout_batch_size` (16 instead of 64).
- Maybe smaller `response_max_len` (128 instead of 256).
- Definitely gradient checkpointing on.

### 2.3 Memory check script (Problem 5.1)

```
python -c "
import torch
from config import GPTConfig
from model import GPT
cfg = GPTConfig()  # gpt2-small defaults
# instantiate 4 models, one dummy forward, print torch.cuda.max_memory_allocated()
"
```

Write the output into this note file for each size that fits.

---

## 3. The outer loop (Problem 5.2)

One PPO iteration has two phases: **rollout** (no-grad, generate data) and **optimize**
(K epochs over that data). Pseudocode:

```
for iter in range(num_iters):

    # --- ROLLOUT PHASE (no grad on policy) ---
    prompts = sample_prompt_batch()
    with torch.no_grad():
        response_ids, logprobs_old, values_old, mask = \
            generate_with_logprobs(policy, value_head, prompts)
        ref_logprobs = gather_logprobs(ref_model(prompts + response_ids), response_ids)
        rm_rewards   = reward_model(prompts + response_ids)  # scalar per row
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

    # --- OPTIMIZE PHASE (grad on policy/value backbone) ---
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

Key invariants:

- **`logprobs_old` is *frozen* at rollout time** and never recomputed across the inner
  epochs. It's the reference point for the importance ratio $\rho_t$.
- **`logprobs_new` is *recomputed* fresh in every minibatch** — that's where the
  policy gradient flows through.
- **All advantages/returns are stop-gradient.** Detach everything coming out of the
  rollout before putting it into `rollout`.

---

## 4. The inner loop (Problem 5.3)

### 4.1 Minibatch sampling

Shuffle a permutation of rollout indices per epoch; iterate in chunks of
`minibatch_size`. Each minibatch spans all batch rows and all time steps — the
"batch dim" for the inner loop is `(rollout_batch_size * T_response)` flattened.

Alternative: slice by row (whole sequences together). Either works; row-wise is
simpler for masking.

### 4.2 K epochs

$K = 4$ is standard. More epochs mean more reuse of each rollout (less generation
cost, more compute) but also more policy drift from $\pi_\text{old}$, which the clip
handles but not indefinitely. If you see "clip fraction" (fraction of tokens where
the min-clip fires) above 40%, lower $K$.

### 4.3 Gradient flow

- Policy gradient enters via `logprobs_new` in `L_pi`.
- Value gradient enters via `values_new` in `L_v`.
- Entropy gradient enters via `logits_new` in `H`.
- All three backprop through the shared backbone. There is no separate step for value
  and policy.

---

## 5. Logging (Problem 5.4)

Log at least these per iteration. A CSV with one row per outer iteration is enough;
emit matplotlib plots every 50 iters.

| Metric             | Expected trajectory                                        |
|--------------------|------------------------------------------------------------|
| `mean_rm_reward`   | Trends up slowly.                                          |
| `mean_kl_k3`       | Trends up slowly from ~0 toward a small positive (~1–6 nats over the response). If it explodes, raise $\beta$ or lower lr. |
| `L_pi`             | Bounces around zero; should not blow up or stall at zero.  |
| `L_v`              | Decreases over time as the value head learns.              |
| `entropy`          | Drops slowly from the SFT entropy baseline.                |
| `clip_fraction`    | Rises from 0 toward 10–30%. >40% means bigger updates than PPO can handle; lower $K$ or lr. |
| `grad_norm`        | Stable at O(1) after clipping. Pre-clip norm may spike.    |
| `tokens_per_sec`   | Roughly constant; drops signal memory pressure.            |
| `response_length`  | Watch for creeping growth (length bias in the RM).         |

### 5.1 Healthy pattern

Reward up, KL up slowly, entropy drifting down slowly, clip fraction ~20%, grad
norm stable, tokens/sec stable.

### 5.2 Unhealthy patterns and remedies

- **Reward up, KL also way up, responses look weird.** Reward hacking. Raise $\beta$
  (the KL coefficient) or lower policy lr. Also check the RM histograms — is there a
  correlation with response length?
- **Reward flat, KL flat.** Policy isn't moving. Lower $\beta$; raise lr.
- **Clip fraction > 50%.** Updates too aggressive for PPO's clip to absorb. Lower
  $K$, lower lr, or smaller minibatches.
- **Entropy crashes toward zero in the first 50 iters.** Premature determinism. Set
  $c_\text{ent} = 0.01$.
- **Grad norm exploding.** Lower lr; check the value loss — usually it's the
  value head overshooting. Make sure value clip $\varepsilon_v$ is on and correct.

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

Smoke test: for each size, instantiate, do one forward + backward at `batch=1,
seq=64`, record `torch.cuda.max_memory_allocated()`. `gpt2-small` and `gpt2-medium`
should step cleanly on a 24GB card; `gpt2-large` and `gpt2-xl` may OOM, which is fine
— the requirement is only that the code compiles and the forward shapes are right,
not that it actually trains at scale.

Write the memory table into this file.

---

## 7. What to commit to `notes/05-ppo.md`

After finishing Module 5, append:

- Memory printout table for each size that fits.
- One training run's CSV (head and tail at least), or plots of reward / KL / entropy
  / clip fraction over iterations.
- One paragraph on instabilities you saw and how you fixed them.
- A sanity check: compare a greedy generation from `rlhf.pt` and from `sft.pt` on
  3 prompts. RLHF should be recognizably different (ideally better).
