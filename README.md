# LearnRLHF — InstructGPT PPO from scratch on GPT-2

From-scratch InstructGPT-style RLHF (Ouyang et al. 2022) on GPT-2 small, trainable on a
single 24GB RTX GPU, pure PyTorch. No `trl`, no `accelerate`, no `transformers.Trainer`.

**Mission:** come out able to derive every forward and backward pass used in SFT, RM, and
PPO on paper, re-implement each component from a blank file, and train a GPT-2 that is
visibly better at instruction following than the raw pretrained checkpoint.

Style target: Karpathy `nanoGPT` + CS231n assignments. Flat `.py` files, minimal
abstractions, aggressive gradient checking, and prose-heavy docstrings.

---

## 1. Setup (once)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/test_grad_check.py -q     # should pass: 3 tests
```

If that last command shows `3 passed`, your grad-check harness works and you can trust
every other gradient check in the repo.

HH-RLHF data is downloaded inside `data_hh.download_hh()` — you implement that in
Problem 0.2.

---

## 2. Scope

**In scope:**
- Pure PyTorch. Only deps: `torch`, `tiktoken`, `datasets`, `numpy`, `matplotlib`, `tqdm`.
- GPT-2 re-implemented (loadable from HuggingFace safetensors) — tied embeddings,
  learned positional embeddings, LayerNorm w/ affine params, GELU.
- SFT, Reward Model, PPO — all three phases of InstructGPT.
- bf16 mixed precision, gradient accumulation, gradient checkpointing.
- Single-GPU only; `GPTConfig` must let you instantiate `gpt2-medium/large/xl` by
  changing `n_layer/n_head/n_embd` — even if training them OOMs.

**Out of scope:**
- DDP/FSDP, ZeRO, DeepSpeed, LoRA/PEFT, Flash-Attn kernels. Use
  `torch.nn.functional.scaled_dot_product_attention` as the one allowed cheat.
- Tokenizer training (use `tiktoken` GPT-2 BPE).
- Human labeling UI.
- Evaluation with GPT-4-as-judge.

---

## 3. Repo layout

```
config.py            # GPTConfig, TrainConfig (SFT, RM, PPO) dataclasses
model.py             # GPT class, load_gpt2_from_hf(), generate()
tokenizer.py         # tiktoken wrapper + chat template helpers
data_hh.py           # HH-RLHF download, tokenize, chat-format, dataloaders
train_sft.py         # SFT loop (prompt-masked CE loss)
train_rm.py          # Reward model loop (Bradley-Terry pairwise loss)
train_ppo.py         # PPO loop (rollout -> advantages -> K epochs of minibatch PPO)
ppo_core.py          # generate_with_logprobs, compute_kl, gae, ppo_loss, value_loss
grad_check.py        # centered-difference helpers; fp64 refs for each loss
eval.py              # side-by-side generation: base vs SFT vs RLHF
tests/
  test_model.py
  test_grad_sft.py
  test_grad_rm.py
  test_grad_ppo.py
notes/               # your derivations, one .md per module
  01-gpt2.md
  02-sft.md
  03-rm.md
  04-ppo-gae.md
  04-ppo-policy.md
  05-ppo.md
  06-eval.md
```

---

## 4. Difficulty estimate

**Overall: Hard.** Comparable to ~2.5 CS231n assignments back-to-back.

| Phase | Difficulty | Why |
|---|---|---|
| GPT-2 re-impl | Medium | Well-trodden; `nanoGPT` is a reference. |
| SFT | Easy-Medium | Biggest gotcha is loss-masking the prompt tokens correctly. |
| Reward Model | Medium | Bradley-Terry loss + last-non-pad-token pooling + numerical stability. |
| PPO | Hard | 4 models in memory, 6 interacting losses/terms, KL schedule, advantage estimation, entropy. |
| Gradient checks | Medium | Centered-difference at fp64 vs. autograd; mask the right pieces. |

Total focused time estimate: **~18-22 hours** for a strong ML engineer who already knows
transformers. Budget more if you're simultaneously learning PPO.

---

## 5. The daily loop (do this for EVERY problem)

For problem `M.P`:

1. **Read.** Find problem `M.P` in the curriculum below (Section 7). Read the `# TODO(M.P):`
   block in the skeleton file — the docstring above it states the math and what the tests check.
2. **Derive.** If the problem has a backward pass (SFT, BT, PPO, KL, GAE, value, entropy),
   write the derivation in `notes/0M-<topic>.md` *before* typing code. Paraphrase nothing —
   if you can't derive it on paper, you don't understand it yet.
3. **Implement** the `TODO`.
4. **Test.** Run only the test(s) for this problem:
   ```bash
   pytest tests/test_grad_ppo.py::test_ppo_policy_loss_grad -q
   ```
5. **Green means green.** If rel_err is `1e-3` on a loss that should be `1e-6`, your analytic
   grad is *almost* right, which is worse than completely wrong. Find the off-by-one / sign /
   mask / shift before continuing.
6. **Commit** with `git commit -m "M.P: <one line>"`. Small commits make bisecting trivial
   when a later module breaks an earlier one.

When in doubt: run `pytest tests/ -q` and work on the FIRST failing test.

---

## 6. Backward passes you must be able to derive

Before coding each loss, produce the derivation on paper and save it in the corresponding
`notes/` file.

**1. Causal LM / SFT.**
Loss = -(1/N_resp) * sum_t mask_t * log softmax(Wh_t)[y_t].
dL/dlogits_t = (softmax(logits_t) - onehot(y_t)) * mask_t / N_resp.

**2. Reward model (Bradley-Terry).**
L = -log sigma(r_c - r_r) = softplus(r_r - r_c).
dL/dr_c = -sigma(r_r - r_c) = sigma(r_c - r_r) - 1.
dL/dr_r =  sigma(r_r - r_c). Note the symmetry.

**3. PPO clipped surrogate.** For each token t:
ratio_t = exp(logpi_t - logpi_old_t).
L_clip_t = -min(ratio_t * A_t, clip(ratio_t, 1-eps, 1+eps) * A_t).
When the clip is active AND the clipped branch is the safer one, gradient through ratio
is zero. When inactive: dL/dlogpi_t = -A_t * ratio_t. Sign flips with sign(A_t).

**4. KL penalty (per-token, used in reward shaping).**
kl_t ~= logpi_t - logpi_ref_t (estimator k1 — unbiased, high-variance). InstructGPT
uses k1. Token reward: r_RM * 1{t=last} - beta * kl_t. Gradient through pi
(ref frozen): d(kl_t)/d(logpi_t) = 1.
k3 = (ratio - 1) - log(ratio) is bias-reduced for logging but nonlinear in ratio.

**5. Value loss.**
L_V = 0.5 * (V_t - R_t)^2 or clipped variant.
R_t is a stop-grad constant even though it was computed from V via GAE.

**6. Entropy bonus.** H = -sum p log p.
dH/dlogits = p * (H - (-log p)). Prevents mode collapse; start at coeff 0.

---

## 7. Curriculum

You MUST go in order — later problems import earlier ones. Each problem is ~30 min of
focused work. Each ends with a concrete artifact (file, test, or plot) and must not
start until the previous one is green.

### Module 0 — Setup (~1 hr)

**0.1 Clean slate.**
Move old files to `old/`, rewrite `requirements.txt` (`torch`, `tiktoken`, `datasets`,
`numpy`, `matplotlib`, `tqdm`), create the file layout above with empty modules.
Artifact: repo structure matches Section 3.

**0.2 HH download + inspection.**
Script that pulls `Anthropic/hh-rlhf`, prints distribution of turn counts,
token-length percentiles (p50/p95/p99) of chosen/rejected, and 3 random samples.
Artifact: `notes/00-data.md` with the output.

---

### Module 1 — GPT-2 from scratch (~4 hrs, 8 problems)

**1.1 Config + embeddings.**
`GPTConfig`, `token_embed + position_embed`. Manual shape assertions.
Artifact: tiny forward test — random ids -> (B, T, C).

**1.2 LayerNorm by hand.**
Write LayerNorm with affine params; gradient-check against `torch.nn.LayerNorm` on fp64
inputs.
Artifact: test passes.

**1.3 Causal self-attention, no flash.**
`q, k, v` from a single `c_attn` linear, reshape, scaled dot product, causal mask,
softmax, weighted sum, `c_proj`. Gradient-check the whole block.
Artifact: test passes.

**1.4 MLP (GELU, 4x expansion).**
Grad-check. Note exact vs. tanh approx — GPT-2 uses exact GELU.
Artifact: test passes.

**1.5 Transformer block.**
Pre-LN: `x + attn(ln1(x))`, `x + mlp(ln2(x))`. Grad-check.
Artifact: test passes.

**1.6 Full GPT-2.**
Stack blocks, final LN, tie `lm_head.weight = wte.weight`. Forward gives `(B, T, V)` logits.
Artifact: forward shape check.

**1.7 Load HF weights.**
Download `gpt2` safetensors, map parameter names (`h.{i}.attn.c_attn` etc.), transpose
Conv1D weights (HF uses Conv1D not Linear for `c_attn/c_proj/c_fc`). Assert your model's
outputs match HF's forward pass to `max_abs_diff < 1e-4` on a fixed prompt.
Artifact: `notes/01-gpt2.md` with the name-map table; parity test passes.

**1.8 Sampling.**
Temperature + top-k + top-p `generate()`, EOS handling, no KV cache yet.
Artifact: generates coherent text from a fixed prompt.

---

### Module 2 — SFT (~2.5 hrs, 5 problems)

**2.1 Chat formatter + tokenizer wrapper.**
Convert HH multi-turn strings into the `<|im_start|>...<|im_end|>` template. Return
`input_ids` and a `loss_mask` that is 1 only on assistant content tokens (inclusive of
their `<|im_end|>`). Unit test: hand-craft a 2-turn example and assert mask positions.
Artifact: test_tokenizer.py passes.

**2.2 Masked causal-LM loss.**
Function `sft_loss(logits, labels, loss_mask)`. Write the derivation in `notes/02-sft.md`
first. Gradient-check w.r.t. logits at fp64 on toy tensors; include the "flip a masked
token, loss unchanged" test.
Artifact: test_grad_sft.py passes.

**2.3 SFT DataLoader.**
Pad to longest-in-batch, return `input_ids`, `labels` (shifted), `loss_mask`,
`attention_mask`. No `ignore_index=-100` hacks — multiply by the mask explicitly.
Verify shapes and dtypes.
Artifact: dataloader shape/dtype test passes.

**2.4 SFT training loop.**
AdamW (beta=(0.9, 0.95), wd=0.1, no-decay on 1D params/bias), cosine LR to 10%,
warmup 200 steps, gradient clipping 1.0, bf16 autocast, grad accumulation. Log train
loss every step, eval loss every 250 steps. Run >= 2 epochs on HH. Save `sft.pt`.
Artifact: `sft.pt` exists; loss curve saved to `runs/`.

**2.5 Qualitative SFT eval.**
`eval.py` mode that loads base vs. SFT and generates on 10 held-out prompts side-by-side.
Write observations in `notes/02-sft.md`. Expect: SFT follows instructions more, base drifts.
Artifact: side-by-side output pasted into notes.

---

### Module 3 — Reward Model (~2.5 hrs, 5 problems)

**3.1 RM architecture.**
`GPT` backbone + `nn.Linear(n_embd, 1)` head. Initialize from `sft.pt` (not base).
Forward returns per-token scalar; reward = value at index of last non-pad token.
Artifact: shape test passes.

**3.2 Preference dataset.**
For each HH pair, tokenize `prompt + chosen` and `prompt + rejected` separately. Return
both with attention masks and last-token indices.
Artifact: dataset shape/dtype test passes.

**3.3 Bradley-Terry loss + derivation.**
`L = -log sigma(r_c - r_r)`. Derive dL/dr_c, dL/dr_r in `notes/03-rm.md`, including the
`softplus` form for numerical stability. Gradient-check at fp64.
Artifact: test_grad_rm.py passes.

**3.4 RM training loop.**
Lower LR than SFT (1e-5 typical), 1 epoch over pairs, eval = pairwise accuracy
(`r_c > r_r`) on held-out. Target: >= 65% pairwise accuracy. Save `rm.pt`.
Artifact: `rm.pt` exists; pairwise accuracy logged.

**3.5 Reward calibration sanity.**
Plot histograms of `r_c` and `r_r` on eval; they should overlap but `r_c` mean > `r_r`
mean. Note it in `notes/03-rm.md`.
Artifact: histogram plot saved.

---

### Module 4 — PPO building blocks (~4 hrs, 8 problems)

Each function lives in `ppo_core.py` and has its own gradient test.

**4.1 Rollout: `generate_with_logprobs(policy, prompts, max_new_tokens)`.**
Returns `response_tokens`, `logprobs_old`, `values_old`, `attention_mask`, `response_mask`.
Single forward per step (no KV cache). Critical: log-prob at position t must be
`log p(token_{t+1} | prefix up to t)`.
Artifact: shape/alignment test passes.

**4.2 Per-token KL.**
`kl_k1 = logprobs - ref_logprobs` (shape `[B, T_resp]`). Include
`kl_k3 = (ratio - 1) - logratio` as the logging estimator. Explain the variance tradeoff
in `notes/04-ppo-gae.md`.
Artifact: unit test that k1 integrates to expected KL on a known distribution.

**4.3 Reward shaping.**
Final-step reward `r_RM` from RM at `<|im_end|>` position; per-token reward
`r_t = -beta * kl_t + r_RM * 1{t = last_response_token}`. Test: with beta->0,
collapses to pure RM terminal reward.
Artifact: unit test passes.

**4.4 GAE.**
Given `r_t, V_t, V_{t+1}` and a done-mask, compute advantages
`A_t = sum (gamma*lambda)^k * delta_{t+k}`, `delta_t = r_t + gamma * V_{t+1} - V_t`.
Backward loop in Python is fine. Returns = A + V (for value target). Unit-test against a
hand-computed 3-step example. Derive in `notes/04-ppo-gae.md`.
Artifact: test passes with hand-computed example.

**4.5 PPO clipped policy loss.**
`L_pi = -E[ min(r*A, clip(r, 1+-eps)*A) * mask ]`. Derive the piecewise gradient in
`notes/04-ppo-policy.md`. Gradient-check at fp64. Include an edge test where every ratio
is clipped — gradient should be zero on those tokens.
Artifact: test_grad_ppo.py::test_ppo_policy_loss_grad passes.

**4.6 Value loss (clipped).**
`L_V = 0.5 * max((V-R)^2, (clip(V, V_old +- eps_v) - R)^2) * mask`. Gradient-check.
Explain why clipping value helps early training in `notes/04-ppo-policy.md`.
Artifact: test_grad_ppo.py::test_value_loss_grad passes.

**4.7 Entropy bonus.**
Mean of `-sum p*log(p)` over masked response tokens. Small coeff (`c_ent ~= 0.0-0.01`);
start at 0 and add if you see premature determinism. Grad check.
Artifact: test_grad_ppo.py::test_entropy_grad passes.

**4.8 Advantage normalization + masking utilities.**
Per-batch normalize advantages (subtract mean, divide std) over *valid* tokens only.
Getting the mask right here is the single most common PPO bug. Write a unit test that
pads half the batch with garbage and verifies stats are unchanged.
Artifact: test passes with padded batch.

---

### Module 5 — PPO training loop (~3 hrs, 5 problems)

**5.1 Model layout and memory map.**
Script that instantiates policy, value (shared backbone with policy, separate value head),
frozen ref, frozen RM. Prints param counts and expected bf16 memory.
`torch.cuda.max_memory_allocated` after one dummy forward must be within 24GB budget.
Artifact: memory printout in `notes/05-ppo.md`.

**5.2 Outer loop (rollout phase).**
For `iter` in range(N): sample a batch of prompts, generate responses, compute ref
logprobs (no grad), compute RM reward (no grad), compute GAE, store in a dict of tensors.
Artifact: one rollout iteration runs without error.

**5.3 Inner loop (optimize phase).**
For K=4 epochs, iterate minibatches of stored rollouts, compute
`L = L_pi + c_v * L_V - c_ent * H`, backward, clip to 1.0, step. Recompute `logpi` and
`V` freshly each minibatch; `logpi_old` is frozen from rollout.
Artifact: one full inner loop iteration runs without error.

**5.4 Logging (fail loudly if these diverge).**
Per-iter log: mean reward, mean KL (k3), policy loss, value loss, entropy, clip fraction,
grad norm, tokens/sec. Save a CSV; emit matplotlib plots every 50 iters.
KL should trend up slowly; reward up; entropy down slowly. Reward collapsing while KL
explodes = reward hacking -> lower LR or raise beta.
Artifact: CSV and plots generated after 10 iters.

**5.5 Config scaling.**
Add `GPTConfig.from_name("gpt2-small"|"gpt2-medium"|"gpt2-large"|"gpt2-xl")`. Run a
smoke test that instantiates each and does one forward+backward with batch 1, seq 64 —
OK if large/xl OOM, but small/medium must step. Document peak memory for each in
`notes/05-ppo.md`.
Artifact: smoke test runs; memory table in notes.

---

### Module 6 — Final evaluation & notes (~1.5 hrs, 3 problems)

**6.1 Generation comparison.**
`eval.py` emits a markdown table of 20 held-out prompts x 3 models (base, SFT, RLHF).
Run it, paste into `notes/06-eval.md`.
Artifact: `notes/06-eval.md` with the full table.

**6.2 Win-rate (manual).**
Self-annotate 20 SFT-vs-RLHF pairs blinded. Target: RLHF wins >= 55%. If not, ablate:
lower KL beta, more PPO epochs, longer RM training.
Artifact: win-rate tally in `notes/06-eval.md`.

**6.3 Retrospective.**
One page: what was hard, what you'd do differently, which grad check saved you, where
the old code was wrong.
Artifact: `notes/06-eval.md` retrospective section.

---

## 8. Default hyperparameters

These are starting points, not gospel. Tune on your own runs.

**SFT:** lr=3e-5, bs=64 (via accum), epochs=2, wd=0.1, warmup=200, cosine->10%.

**RM:** lr=1e-5, bs=32, epochs=1, init from `sft.pt`.

**PPO:** lr=1e-6 (policy) / 1e-5 (value head only), rollout batch=64 prompts,
response_len=128-256, K=4, minibatch=16, gamma=1.0 (episodic text), lambda=0.95,
clip_eps=0.2, value_clip=0.2, beta_kl=0.02, entropy_coef=0.0.

---

## 9. Data: Anthropic HH-RLHF

Source: `Anthropic/hh-rlhf` on HuggingFace. Raw format: `{"chosen": str, "rejected": str}`
where both strings are multi-turn dialogues alternating `Human:` / `Assistant:`.

Chat template enforced end-to-end:
```
<|im_start|>user
<turn text><|im_end|>
<|im_start|>assistant
<turn text><|im_end|>
```

Three derived datasets from `data_hh.py`:

1. **SFT:** full chosen dialogue. `loss_mask` = 1 only on assistant tokens (including
   their `<|im_end|>`), 0 on user tokens and prompt scaffolding.
2. **Preference pairs:** `(prompt, chosen_response, rejected_response)` for the RM.
3. **Prompt-only:** prompts truncated so the model has budget to generate. Keep prompts
   <= 512 tok, rollouts <= 256 new tokens.

`<|im_start|>` and `<|im_end|>` are not in base GPT-2 BPE — they are encoded as their
literal UTF-8 bytes and BPE splits them. Don't change this.

---

## 10. Hardware & memory budget (24GB GPU)

GPT-2 small = 124M params. PPO holds four models simultaneously:

| Model | Params | Grads | Optim (AdamW, fp32) |
|---|---|---|---|
| Policy (trainable) | 124M | yes | 2x state |
| Value head + backbone (trainable) | shared | yes | 2x state |
| Reference policy (frozen) | 124M | no | no |
| Reward model (frozen) | 124M | no | no |

Rough budget in bf16 w/ fp32 optimizer state, gradient checkpointing ON:
- 4 x 124M x 2B (bf16 weights) ~= 1.0 GB weights
- Trainable fp32 master copy + Adam m,v: ~124M x 12B ~= 1.5 GB
- Activations (ckpt'd, batch 4 x seq 512): ~4-8 GB
- Rollout buffers: < 1 GB

Fits comfortably. Headroom is for scaling to `gpt2-medium` (355M) — will be tight.

---

## 11. Gradient-check protocol

Every loss you write gets a dedicated test. Template:

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
- **fp64 + tiny dims** (e.g. 2 layers, n_embd=16, 1-2 heads, seq 8, batch 2).
- Check gradient w.r.t. both inputs and parameters on at least one test per loss.
- **Mask test:** when loss has a mask, write a second test that flips a masked-out element
  and asserts the loss is unchanged.
- Don't skip this. PPO in particular silently "trains" with wrong signs and off-by-one
  shifts for thousands of steps.

---

## 12. Debugging tips

- **Gradient checks lie only when you lie to them.** If a grad check passes but training
  does nothing, the bug is in the masking / shift / reduction. Look at `.shape` and
  `.sum()` of every mask.
- **PPO "trains" while being silently broken.** Always watch all six logs simultaneously:
  reward, KL (k3), policy loss, value loss, entropy, clip fraction. Reward up while KL
  explodes = reward hacking -> lower LR or raise beta.
- **When confused about alignment in PPO** (logits[t] vs response[t]), write a 3-token
  example on paper and verify indices before changing code.
- **`old/` is your oracle** for seeing how someone else solved a piece (imperfectly).
  Don't copy, but do diff against it when stuck.

---

## 13. Definition of done

- [ ] `python -m pytest tests/ -q` is fully green.
- [ ] `python eval.py --models base,sft,rlhf --n 20` produces a markdown table where the
      RLHF column is visibly better at instruction-following on most rows.
- [ ] You can derive, on a whiteboard, the backward pass of every loss in the repo.
- [ ] `notes/` contains one markdown per module with your own derivations.
- [ ] Single `config.py` switches model size; `gpt2-small` trains end-to-end on a 24GB GPU.
- [ ] No dependency on `transformers`, `trl`, `accelerate`, `peft`, or `deepspeed`.

---

## 14. Norms

- No `transformers.Trainer`, no `trl`, no `accelerate`, no `peft`, no `deepspeed`.
  `transformers` is only allowed in `load_gpt2_from_hf` for the one-time weight load.
- No `ignore_index=-100`. Always an explicit `loss_mask` tensor, multiplied in.
- `nn.Module` yes; dataclasses yes; anything more abstract — no.
- Comments explain *why*, not *what*.
- No emojis in code or notes.