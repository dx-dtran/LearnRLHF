# LearnRLHF — InstructGPT PPO from scratch on GPT-2

## 0. Mission

Rewrite this repo as a **teaching-grade, from-scratch implementation of InstructGPT-style RLHF**
(Ouyang et al. 2022) on GPT-2 small, trainable on a single 24GB RTX GPU, using the
Anthropic HH-RLHF dataset.

The learner (`daniel@`) must come out able to:

1. Derive every forward and backward pass used across SFT, RM, and PPO on paper.
2. Re-implement each component from a blank file.
3. Explain *why* each loss has the form it does, and what each gradient term means.
4. Train a GPT-2 small that is qualitatively better at instruction following than the raw
   pretrained checkpoint, and know exactly which knobs matter.

Style target: **Karpathy `nanoGPT` / `minGPT` + CS231n assignments**. Functions in flat
`.py` files, minimal abstractions, no frameworks (no `trl`, no `accelerate`, no
`transformers.Trainer`), aggressive gradient checking, and prose-heavy docstrings.

This file is the **curriculum and spec**. Do not implement ahead of the module the learner
is currently on.

---

## 1. Scope & non-goals

In scope:
- Pure PyTorch. Only deps: `torch`, `tiktoken`, `datasets` (for HH download), `numpy`,
  `matplotlib`. `tqdm` allowed. Nothing else.
- GPT-2 re-implemented (loadable from HuggingFace safetensors) — tied embeddings,
  learned positional embeddings, LayerNorm w/ affine params, GELU.
- SFT, Reward Model, PPO — all three phases of InstructGPT.
- bf16 mixed precision, gradient accumulation, gradient checkpointing.
- Single-GPU only, but `GPTConfig` must let you *instantiate* `gpt2-medium/large/xl`
  by changing `n_layer/n_head/n_embd` — even if training them OOMs.

Out of scope:
- DDP/FSDP, ZeRO, DeepSpeed, LoRA/PEFT, Flash-Attn kernels. Use
  `torch.nn.functional.scaled_dot_product_attention` as the one allowed "cheat" for speed.
- Tokenizer training (use `tiktoken` GPT-2 BPE).
- Human labeling UI.
- Evaluation with GPT-4-as-judge (we do qualitative + small-scale win-rate by hand).

---

## 2. Difficulty estimate

**Overall difficulty: Hard.** Comparable to roughly 2.5 CS231n assignments back-to-back.

| Phase | Difficulty | Why |
|---|---|---|
| GPT-2 re-impl | Medium | Well-trodden; `nanoGPT` is a reference. |
| SFT | Easy–Medium | Biggest gotcha is loss-masking the prompt tokens correctly. |
| Reward Model | Medium | Bradley–Terry loss + last-non-pad-token pooling + numerical stability. |
| PPO | **Hard** | 4 models in memory, 6 interacting losses/terms, KL schedule, advantage estimation, entropy. Memory engineering on 24GB. |
| Gradient checks | Medium | Centered-difference at fp64 vs. autograd; mask the right pieces. |

Total focused time estimate: **~18–22 hours** of head-down work for a strong ML engineer
who already knows transformers. Budget more if you're simultaneously learning PPO.

---

## 3. Hardware & memory budget (24GB)

GPT-2 small = 124M params. PPO holds four models:

| Model | Params | Grads | Optim (AdamW, fp32) | bf16 activations? |
|---|---|---|---|---|
| Policy (trainable) | 124M | yes | 2× state | yes + grad ckpt |
| Value head + backbone (trainable, shared or separate) | 124M or shared | yes | 2× state | yes |
| Reference policy (frozen) | 124M | no | no | yes |
| Reward model (frozen) | 124M | no | no | yes |

Rough budget in bf16 w/ fp32 optimizer state, gradient checkpointing ON:
- 4 × 124M × 2B (bf16 weights) ≈ 1.0 GB weights
- Trainable fp32 master copy + Adam m,v: ~124M × 12B ≈ 1.5 GB
- Activations (ckpt'd, batch 4 × seq 512): ~4–8 GB
- Rollout buffers (logprobs, values, advantages, rewards for ~64 × 256 tokens): <1 GB

Fits comfortably. Headroom is for scaling to `gpt2-medium` (355M) — will be tight.

Hyperparameters MUST live in a single `config.py` module so swapping
`gpt2-small ↔ medium ↔ large ↔ xl` is a one-line change. Train is expected to work for
small; larger sizes are "compiles and steps, even if slow/OOM".

---

## 4. Target repo layout (after rewrite)

Flat. No `src/`, no `models/` package — nanoGPT style.

```
config.py            # GPTConfig, TrainConfig (SFT, RM, PPO) dataclasses
model.py             # GPT class, load_gpt2_from_hf(), generate()
tokenizer.py         # tiktoken wrapper + chat template helpers
data_hh.py           # HH-RLHF download, tokenize, chat-format, dataloaders
                     # produces: SFT dataset, preference pairs dataset, prompt-only dataset
train_sft.py         # SFT loop (prompt-masked CE loss)
train_rm.py          # Reward model loop (Bradley–Terry pairwise loss)
train_ppo.py         # PPO loop (rollout → advantages → K epochs of minibatch PPO)
ppo_core.py          # generate_with_logprobs, compute_kl, gae, ppo_loss, value_loss
grad_check.py        # centered-difference helpers; fp64 refs for each loss
tests/
  test_model.py         # forward shapes, weight-load parity vs. HF
  test_grad_sft.py
  test_grad_rm.py
  test_grad_ppo.py      # surrogate, value, entropy, KL
eval.py              # side-by-side generation: base vs SFT vs RLHF
notes/
  01-gpt2.md ...      # one .md per module, derivations + what you learned
```

Preserve nothing from the old repo except `train.jsonl` / `test.jsonl` if they cover HH
(spot check: they do — chosen/rejected token lists with `<|im_start|>` tags). Everything
else — `gpt.py`, `rlhf_ppo.py`, `sft_data.py`, `train_sft.py`, old `.pt` checkpoints —
gets deleted or moved to `old/` before starting.

---

## 5. Data: Anthropic HH-RLHF

Use the `Anthropic/hh-rlhf` split on HuggingFace. Raw format: `{"chosen": str, "rejected": str}`
where both strings are multi-turn dialogues alternating `Human:` / `Assistant:`.

Chat template we'll enforce end-to-end (matches the existing jsonl):

```
<|im_start|>user
<turn text><|im_end|>
<|im_start|>assistant
<turn text><|im_end|>
```

Three derived datasets, produced by `data_hh.py`:
1. **SFT**: single-string = full chosen dialogue. `loss_mask` = 1 only on assistant tokens
   (including their `<|im_end|>`), 0 on user tokens and prompt scaffolding. *This is the
   bug in the current repo — it trains on user turns too.*
2. **Preference pairs**: `(prompt, chosen_response, rejected_response)` for the RM.
3. **Prompt-only**: prompts truncated so the model has budget to generate — for PPO
   rollouts. Keep prompts ≤ 512 tok, rollouts ≤ 256 new tokens.

`<|im_start|>`, `<|im_end|>` are not in base GPT-2 BPE. Either (a) reserve two unused BPE
ids and treat them as special (document which), or (b) encode them as their literal UTF-8
bytes and let BPE split — current data uses (b), stick with that for minimal churn.

---

## 6. Gradient-check protocol

Every loss the learner writes gets a dedicated test. Template:

```python
def test_<loss>_grad():
    torch.manual_seed(0)
    # small dims, fp64
    model = SmallModel(...).double()
    x = torch.randn(..., dtype=torch.float64, requires_grad=True)
    loss = my_loss(model, x, ...)
    loss.backward()
    analytic = x.grad.clone()

    numeric = torch.zeros_like(x)
    eps = 1e-6
    for idx in iter_indices(x):
        x_ = x.detach().clone(); x_[idx] += eps
        lp = my_loss(model, x_, ...).item()
        x_ = x.detach().clone(); x_[idx] -= eps
        lm = my_loss(model, x_, ...).item()
        numeric[idx] = (lp - lm) / (2*eps)

    rel = (analytic - numeric).norm() / (analytic.norm() + numeric.norm() + 1e-12)
    assert rel < 1e-5
```

Rules:
- **fp64 + tiny dims** (e.g. 2 layers, n_embd=16, 1–2 heads, seq 8, batch 2).
- Always check gradient w.r.t. *both* inputs and parameters on at least one test per loss.
- Mask test: when loss has a mask (SFT prompt mask, RM end-of-seq pooling, PPO per-token),
  write a second test that flips a masked-out element and asserts the loss is unchanged.
- Don't skip this — PPO in particular silently "trains" with wrong signs and off-by-one
  shifts for thousands of steps.

---

## 7. Backward-pass mental model (learner should be able to derive these)

The learner should, before coding each loss, be able to produce these on paper:

1. **Causal LM / SFT.** Loss = −(1/N_resp) Σ_t mask_t · log softmax(Wh_t)_{y_t}.
   ∂L/∂logits_t = (softmax(logits_t) − onehot(y_t)) · mask_t / N_resp. This is the one
   free gradient everyone memorizes; derive it anyway.

2. **Reward model (Bradley–Terry).**
   L = −log σ(r_c − r_r) = softplus(r_r − r_c).
   ∂L/∂r_c = −σ(r_r − r_c) = σ(r_c − r_r) − 1.
   ∂L/∂r_r =  σ(r_r − r_c). Note the symmetry.

3. **PPO clipped surrogate.** For each token t:
   ratio_t = exp(logπ_t − logπ_old_t),
   L_clip_t = −min(ratio_t · A_t, clip(ratio_t, 1−ε, 1+ε) · A_t).
   Derive: when the clip is active (ratio outside [1−ε, 1+ε] AND the "safer" branch is the
   clipped one), the gradient through ratio is zero for that token. When inactive,
   ∂L/∂logπ_t = −A_t · ratio_t. Sanity: sign flips with sign(A_t) — that's the policy
   gradient.

4. **KL penalty (per-token, used in reward shaping).**
   kl_t ≈ logπ_t − logπ_ref_t (the "estimator k1" — unbiased but high-variance). InstructGPT
   uses k1. The token reward is `r_RM · 1{t=last}  −  β · kl_t`.
   Gradient through π (ref is frozen): ∂kl_t/∂logπ_t = 1. If using k3 (`(r-1) - logr` with
   r = π/π_ref), note it's bias-reduced but the gradient is nonlinear in ratio — explain
   why k1 is usually fine for the penalty but k3 is preferred for logging.

5. **Value loss.**
   L_V = ½ (V_t − R_t)² or clipped variant ½ max((V_t−R_t)², (clip(V_t, V_old±ε)−R_t)²).
   ∂L/∂V_t is trivial; the interesting part is showing GAE's R_t is treated as a constant
   (stop-grad) even though it was computed from V itself.

6. **Entropy bonus.** H = −Σ p log p, ∂H/∂logits = p ⊙ (H − (−logp)). Usually added with
   a small coefficient to prevent mode collapse.

Each `notes/NN-*.md` should contain the learner's own derivation before the code lands.

---

## 8. Curriculum — 30-minute problems

Each problem is ~30 min of focused work (some 15, some 45 — the unit is "one sitting").
Problems numbered within a module: `M.P`. **Do them in order.** Each problem ends with a
concrete artifact (file, test, or plot) and must not start until the previous one is
green.

### Module 0 — Setup (≈1 hr)

- **0.1 Clean slate.** Move old files to `old/`, rewrite `requirements.txt`
  (`torch`, `tiktoken`, `datasets`, `numpy`, `matplotlib`, `tqdm`), create the file layout
  above with empty modules.
- **0.2 HH download + inspection.** Script that pulls Anthropic/hh-rlhf, prints
  distribution of turn counts, token-length percentiles (p50/p95/p99) of chosen/rejected,
  and 3 random samples. Save to `notes/00-data.md`.

### Module 1 — GPT-2 from scratch (≈4 hrs, 8 problems)

- **1.1 Config + embeddings.** `GPTConfig`, `token_embed + position_embed`. Manual shape
  assertions. Tiny forward test: random ids → (B,T,C).
- **1.2 LayerNorm by hand.** Write LayerNorm with affine params; gradient-check against
  `torch.nn.LayerNorm` on fp64 inputs.
- **1.3 Causal self-attention, no flash.** `q,k,v` from a single `c_attn` linear, reshape,
  scaled dot product, causal mask, softmax, weighted sum, `c_proj`. Gradient-check the
  whole block.
- **1.4 MLP (GELU, 4× expansion).** Grad-check. Note exact vs. tanh approx — GPT-2 uses
  exact `gelu`.
- **1.5 Transformer block.** Pre-LN: `x + attn(ln1(x))`, `x + mlp(ln2(x))`. Grad-check.
- **1.6 Full GPT-2.** Stack blocks, final LN, tie `lm_head.weight = wte.weight`. Forward
  gives `(B, T, V)` logits.
- **1.7 Load HF weights.** Download `gpt2` safetensors, map parameter names
  (`h.{i}.attn.c_attn` etc.), **transpose Conv1D weights** (HF uses Conv1D not Linear for
  `c_attn/c_proj/c_fc`). Assert your model's outputs match HF's forward pass to
  `max_abs_diff < 1e-4` on a fixed prompt. Save `notes/01-gpt2.md` with the name-map table.
- **1.8 Sampling.** Temperature + top-k + top-p `generate()`, EOS handling, **no KV cache
  yet** — add cache only if PPO rollouts are too slow (keep the code readable first).

### Module 2 — SFT (≈2.5 hrs, 5 problems)

- **2.1 Chat formatter + tokenizer wrapper.** Convert HH multi-turn strings into the
  `<|im_start|>…<|im_end|>` template. Return `input_ids` and a `loss_mask` that is 1 only
  on assistant content tokens (inclusive of their `<|im_end|>`). Unit test: hand-craft a
  2-turn example and assert mask positions.
- **2.2 Masked causal-LM loss.** Function `sft_loss(logits, labels, loss_mask)`. Derivation
  in `notes/02-sft.md`. Gradient-check w.r.t. logits at fp64 on toy tensors; include the
  "flip a masked token, loss unchanged" test.
- **2.3 SFT DataLoader.** Pad to longest-in-batch, return `input_ids`, `labels` (shifted),
  `loss_mask`, `attention_mask`. No `ignore_index=-100` hacks — just multiply by the mask,
  explicitly. Verify shapes and dtypes.
- **2.4 SFT training loop.** AdamW (β=(0.9,0.95), wd=0.1, no-decay on 1D params/bias),
  cosine LR to 10%, warmup 200 steps, gradient clipping 1.0, bf16 autocast, grad
  accumulation. Log train loss every step, eval loss every 250 steps. Run ≥2 epochs on HH.
  Save `sft.pt`.
- **2.5 Qualitative SFT eval.** `eval.py` mode that loads base vs. SFT and generates on 10
  held-out prompts side-by-side. Write observations in `notes/02-sft.md`. Expect: SFT
  follows instructions more, base drifts.

### Module 3 — Reward Model (≈2.5 hrs, 5 problems)

- **3.1 RM architecture.** `GPT` backbone + `nn.Linear(n_embd, 1)` head. Initialize from
  `sft.pt` (not base). Forward returns per-token scalar; reward = value at index of last
  non-pad token.
- **3.2 Preference dataset.** For each HH pair, tokenize `prompt + chosen` and
  `prompt + rejected` separately (share the prompt prefix in docs but not in code —
  clarity over perf). Return both with attention masks and last-token indices.
- **3.3 Bradley–Terry loss + derivation.** `L = -log σ(r_c - r_r)`. Derive ∂L/∂r_c, ∂L/∂r_r
  in `notes/03-rm.md`, including the `softplus` form for numerical stability. Gradient-check
  at fp64.
- **3.4 RM training loop.** Lower LR than SFT (1e-5 typical), 1 epoch over pairs, eval =
  pairwise accuracy (`r_c > r_r`) on held-out. Target: ≥ 65% pairwise accuracy. Save
  `rm.pt`.
- **3.5 Reward calibration sanity.** Plot histograms of `r_c` and `r_r` on eval; they
  should overlap a lot but `r_c` mean > `r_r` mean. Note it in `notes/03-rm.md`.

### Module 4 — PPO building blocks (≈4 hrs, 8 problems)

Each of these is its own file-level function in `ppo_core.py`, with its own gradient test.

- **4.1 Rollout: `generate_with_logprobs(policy, prompts, max_new_tokens)`.** Returns
  `response_tokens`, `logprobs_old`, `values_old`, `attention_mask`, `response_mask`. Do it
  with a single forward per step (no KV cache unless Module 1.8 added one). Correctness is
  critical here: the log-prob at position t must be `log p(token_{t+1} | prefix up to t)`.
- **4.2 Per-token KL.** `kl_k1 = logprobs - ref_logprobs` (shape `[B, T_resp]`). Include
  `kl_k3 = (ratio - 1) - logratio` as the logging estimator. Explain the variance tradeoff.
- **4.3 Reward shaping.** Final-step reward `r_RM` from RM at `<|im_end|>` position;
  per-token reward `r_t = -β · kl_t + r_RM · 1{t = last_response_token}`. Test with
  β→0 collapses to pure RM terminal reward.
- **4.4 GAE.** Given `r_t, V_t, V_{t+1}` and a done-mask, compute advantages
  `A_t = Σ (γλ)^k δ_{t+k}`, `δ_t = r_t + γ V_{t+1} - V_t`. Backward loop in Python is fine.
  Returns = A + V (for value target). Unit-test against a hand-computed 3-step example.
  Derive in `notes/04-ppo-gae.md`.
- **4.5 PPO clipped policy loss.** `L_π = -E[ min(r A, clip(r, 1±ε) A) · mask ]`. Derive the
  piecewise gradient in `notes/04-ppo-policy.md`. Gradient-check at fp64. Include an edge
  test where every ratio is clipped — gradient should be zero on those tokens.
- **4.6 Value loss (clipped).** `L_V = 0.5 · max((V − R)², (clip(V, V_old ± ε_v) − R)²) · mask`.
  Gradient-check. Explain why clipping value helps early training (prevents value-head
  overshoot from dominating).
- **4.7 Entropy bonus.** Mean of `-Σ p logp` over masked response tokens. Small coeff
  (`c_ent ≈ 0.0–0.01`); start at 0 and add if you see premature determinism. Grad check.
- **4.8 Advantage normalization + masking utilities.** Per-batch normalize advantages
  (subtract mean, divide std) over *valid* tokens only. Getting the mask right here is the
  single most common PPO bug — write a unit test that pads half the batch with garbage and
  verifies stats are unchanged.

### Module 5 — PPO training loop (≈3 hrs, 5 problems)

- **5.1 Model layout and memory map.** Script that instantiates policy, value (share
  backbone with policy OR separate — document choice; shared is cheaper, separate is more
  stable — default: shared backbone, separate value head), frozen ref, frozen RM. Prints
  param counts and expected bf16 memory. `torch.cuda.max_memory_allocated` after one dummy
  forward must be within budget.
- **5.2 Outer loop (rollout phase).** For `iter` in range(N): sample a batch of prompts,
  generate responses, compute ref logprobs (no grad), compute RM reward (no grad), compute
  GAE, store everything in a dict of tensors.
- **5.3 Inner loop (optimize phase).** For `K=4` epochs, iterate minibatches of stored
  rollouts, compute `L = L_π + c_v · L_V − c_ent · H`, backward, clip to 1.0, step.
  Recompute `logπ` and `V` freshly each minibatch; `logπ_old` is frozen from rollout.
- **5.4 Logging (fail loudly if these diverge).** Per-iter log: mean reward, mean KL (k3),
  policy loss, value loss, entropy, clip fraction, grad norm, tokens/sec. Save a CSV,
  emit matplotlib plots every 50 iters. KL should trend up slowly; reward up; entropy down
  slowly. Reward collapsing while KL explodes = reward hacking → lower LR or raise β.
- **5.5 Config scaling.** Add `GPTConfig.from_name("gpt2-small"|"gpt2-medium"|"gpt2-large"|
  "gpt2-xl")`. Run a smoke test that instantiates each and does one forward+backward with
  batch 1, seq 64 — OK if large/xl OOM, but small/medium must step. Document peak memory
  for each in `notes/05-ppo.md`.

### Module 6 — Final evaluation & notes (≈1.5 hrs, 3 problems)

- **6.1 Generation comparison.** `eval.py` emits a markdown table of 20 held-out prompts ×
  3 models (base, SFT, RLHF). Run it, paste into `notes/06-eval.md`.
- **6.2 Win-rate (manual).** Self-annotate 20 SFT-vs-RLHF pairs blinded. Target: RLHF wins
  ≥ 55%. If not, ablate: lower KL β, more PPO epochs, longer RM training.
- **6.3 Retrospective.** One page: what was hard, what you'd do differently, which grad
  check saved you, where the old code was wrong.

---

## 9. Default hyperparameters (starting point, not gospel)

SFT: lr=3e-5, bs=64 (via accum), epochs=2, wd=0.1, warmup=200, cosine→10%.

RM: lr=1e-5, bs=32, epochs=1, init from `sft.pt`.

PPO: lr=1e-6 (policy) / 1e-5 (value head only, if separate), rollout batch=64 prompts,
response_len=128–256, K=4, minibatch=16, γ=1.0 (episodic text), λ=0.95, clip_ε=0.2,
value_clip=0.2, β_kl=0.02 (adaptive target-KL=6 nat over 256 tok — optional), entropy_coef=0.0.

These match common InstructGPT-era settings; tune on your own runs.

---

## 10. Definition of done

The repo ships when **all** of these hold:

- [ ] `python -m pytest tests/` green — every loss has a gradient check.
- [ ] `python eval.py --compare base,sft,rlhf --n 20` produces the side-by-side table and
      the RLHF column is *visibly* better at instruction following on a majority of prompts.
- [ ] Learner can, on a whiteboard, derive: SFT grad, BT grad, PPO clipped-ratio grad,
      GAE recursion, KL k1 vs k3, value clip, entropy bonus.
- [ ] `notes/` contains one markdown per module with the learner's own derivations — not
      Claude's prose.
- [ ] Single `config.py` switches model size; `gpt2-small` trains end-to-end on a 24GB GPU.
- [ ] No dependency on `transformers`, `trl`, `accelerate`, `peft`, or `deepspeed`.

---

## 11. Working norms for Claude inside this repo

- **Do not write ahead.** When the learner is on 2.3, do not produce 2.4. They learn by
  writing.
- **One loss, one file, one grad check.** Any new loss function lands with its test in the
  same PR/turn.
- **Never use `ignore_index=-100` tricks**; always an explicit `loss_mask` tensor
  multiplied in. Masking bugs are the whole point of this course.
- **No emojis in code or notes.**
- **No new abstractions.** `nn.Module` is allowed, dataclasses are allowed, anything
  fancier (registries, configs-of-configs, training framework) gets rejected.
- **Prefer `torch.testing.assert_close` with explicit atol/rtol** over hand-rolled
  tolerances, except in grad-check code where you want to see the relative error.
- **When stuck, write the derivation on paper first.** If Claude is writing a PPO surrogate
  and can't state the gradient closed-form, stop and derive, don't guess.