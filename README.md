# LearnRLHF — InstructGPT PPO from scratch on GPT-2

From-scratch InstructGPT-style RLHF (Ouyang et al. 2022) on GPT-2 small, single 24GB GPU,
pure PyTorch. No `trl`, no `accelerate`, no `transformers.Trainer`.

By the end you should be able to derive every forward and backward pass in SFT, RM, and PPO
on paper, re-implement each component from a blank file, and produce a GPT-2 that is visibly
better at instruction following than the raw pretrained checkpoint.

Style: Karpathy `nanoGPT` + CS231n assignments. Flat `.py` files, minimal abstractions,
aggressive gradient checking, prose-heavy docstrings.

---

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/test_grad_check.py -q     # should pass: 3 tests
```

HH-RLHF data is downloaded inside `data_hh.download_hh()` — you implement that in Problem 0.2.

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
notes/               # theory references (read before coding each module)
  00-data.md
  01-gpt2.md
  02-sft.md
  03-rm.md
  04-ppo-gae.md
  04-ppo-kl.md
  04-ppo-policy.md
  05-ppo.md
  06-eval.md
```

---

## 4. Per-problem workflow

For each problem `M.P`:

1. **Read the theory.** Open the relevant `notes/0M-*.md` file and read the derivation
   for the loss / module you're about to implement. These notes are pre-written theory
   references; they contain every equation and gradient you'll need. Re-derive the
   backward passes on paper as you read — if you can't, re-read.
2. **Read** the `# TODO(M.P):` block in the skeleton file. The docstring states the
   math and what the tests check.
3. **Implement** the `TODO`.
4. **Test.** Run only the test(s) for this problem:
   ```bash
   pytest tests/test_grad_ppo.py::test_ppo_policy_loss_grad -q
   ```
5. **Green means green.** If rel_err is `1e-3` on a loss that should be `1e-6`, find
   the off-by-one / sign / mask / shift before continuing.
6. **Annotate.** Each `notes/` file has a "What to commit" section at the end —
   append your run outputs (loss curves, hparams you changed, surprises) there as you
   go. Future-you will want them.
7. **Commit** with `git commit -m "M.P: <one line>"`.

When in doubt: `pytest tests/ -q` and fix the first failing test.

---

## 5. Backward passes

Short summaries. Full derivations live in the corresponding `notes/` file — read the
note before implementing the loss, and re-derive each gradient on paper as you go.

**1. Causal LM / SFT.**

$$
\mathcal{L} = -\frac{1}{N_\text{resp}} \sum_t m_t \, \log \mathrm{softmax}(W h_t)_{y_t}
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathrm{logits}_t}
= \frac{m_t}{N_\text{resp}} \left( \mathrm{softmax}(\mathrm{logits}_t) - \mathbf{1}_{y_t} \right)
$$

**2. Reward model (Bradley–Terry).**

$$
\mathcal{L} = -\log \sigma(r_c - r_r) = \mathrm{softplus}(r_r - r_c)
$$

$$
\frac{\partial \mathcal{L}}{\partial r_c} = \sigma(r_c - r_r) - 1,
\qquad
\frac{\partial \mathcal{L}}{\partial r_r} = \sigma(r_r - r_c)
$$

Note the symmetry: the gradients sum to zero.

**3. PPO clipped surrogate.**

$$
\rho_t = \exp(\log \pi_t - \log \pi_t^\text{old})
$$

$$
\mathcal{L}_t^\text{clip}
= -\min\!\left( \rho_t A_t, \; \mathrm{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon) \cdot A_t \right)
$$

When the clip is active and the clipped branch is the one that would raise the
objective, $\partial \mathcal{L} / \partial \log \pi_t = 0$.
When inactive: $\partial \mathcal{L} / \partial \log \pi_t = -A_t \rho_t$. Sign flips
with $\mathrm{sign}(A_t)$.

**4. KL penalty (per-token).**

$$
\mathrm{KL}_t \approx \log \pi_t - \log \pi_t^\text{ref}
\qquad (k_1 \text{ estimator, unbiased, high variance})
$$

Token reward: $r_t = r_\text{RM} \cdot \mathbf{1}_{t=T} - \beta \cdot \mathrm{KL}_t$.
Since $\pi^\text{ref}$ is frozen: $\partial \mathrm{KL}_t / \partial \log \pi_t = 1$.

The $k_3$ estimator $(\rho - 1) - \log \rho$ is nonnegative and lower-variance, preferred
for logging but nonlinear in $\rho$ as a penalty.

**5. Value loss.**

$$
\mathcal{L}_V = \tfrac{1}{2}(V_t - R_t)^2
\qquad \text{(or clipped variant; see notes/04-ppo-policy.md)}
$$

$R_t$ is a stop-gradient constant even though it was computed from $V$ via GAE.

**6. Entropy bonus.**

$$
H = -\sum_a \pi(a) \, \log \pi(a)
$$

$$
\frac{\partial H}{\partial \mathrm{logits}}
= -\, \pi \odot (\log \pi + H)
$$

Start with coefficient 0; increase if you see premature determinism.

---

## 6. Curriculum

Problems must be done in order — later ones import earlier ones. Each ends with a concrete
artifact (file, test, or plot) and does not start until the previous one is green.

### Module 0 — Setup

**0.1 Clean slate.**
Rewrite `requirements.txt` (`torch`, `tiktoken`, `datasets`,
`numpy`, `matplotlib`, `tqdm`), create the file layout above with empty modules.
Artifact: repo structure matches Section 3.

**0.2 HH download + inspection.**
Pull `Anthropic/hh-rlhf`, print distribution of turn counts, token-length percentiles
(p50/p95/p99) of chosen/rejected, and 3 random samples.
Artifact: `notes/00-data.md` with the output.

---

### Module 1 — GPT-2 from scratch

**1.1 Config + embeddings.**
`GPTConfig`, `token_embed + position_embed`. Manual shape assertions.
Artifact: tiny forward test — random ids → (B, T, C).

**1.2 LayerNorm by hand.**
Write LayerNorm with affine params; gradient-check against `torch.nn.LayerNorm` on fp64.
Artifact: test passes.

**1.3 Causal self-attention, no flash.**
`q, k, v` from a single `c_attn` linear, reshape, scaled dot product, causal mask,
softmax, weighted sum, `c_proj`. Gradient-check the whole block.
Artifact: test passes.

**1.4 MLP (GELU, 4× expansion).**
Grad-check. Note exact vs. tanh approx — GPT-2 uses exact GELU.
Artifact: test passes.

**1.5 Transformer block.**
Pre-LN: `x + attn(ln1(x))`, `x + mlp(ln2(x))`. Grad-check.
Artifact: test passes.

**1.6 Full GPT-2.**
Stack blocks, final LN, tie `lm_head.weight = wte.weight`. Forward gives `(B, T, V)` logits.
Artifact: forward shape check.

**1.7 Load HF weights.**
Download `gpt2` safetensors, map parameter names, transpose Conv1D weights (HF uses Conv1D
not Linear for `c_attn/c_proj/c_fc`). Assert outputs match HF to `max_abs_diff < 1e-4`.
Artifact: `notes/01-gpt2.md` with the name-map table; parity test passes.

**1.8 Sampling.**
Temperature + top-k + top-p `generate()`, EOS handling, no KV cache yet.
Artifact: generates coherent text from a fixed prompt.

---

### Module 2 — SFT

**2.1 Chat formatter + tokenizer wrapper.**
Convert HH multi-turn strings into the `<|im_start|>...<|im_end|>` template. Return
`input_ids` and `loss_mask` = 1 only on assistant content tokens (inclusive of `<|im_end|>`).
Unit test: hand-craft a 2-turn example and assert mask positions.
Artifact: test_tokenizer.py passes.

**2.2 Masked causal-LM loss.**
`sft_loss(logits, labels, loss_mask)`. Derive in `notes/02-sft.md` first.
Gradient-check w.r.t. logits at fp64; include the "flip a masked token, loss unchanged" test.
Artifact: test_grad_sft.py passes.

**2.3 SFT DataLoader.**
Pad to longest-in-batch, return `input_ids`, `labels` (shifted), `loss_mask`,
`attention_mask`. No `ignore_index=-100` — multiply by the mask explicitly.
Artifact: shape/dtype test passes.

**2.4 SFT training loop.**
AdamW (β=(0.9, 0.95), wd=0.1, no-decay on 1D params/bias), cosine LR to 10%,
warmup 200 steps, grad clipping 1.0, bf16 autocast, grad accumulation.
Log train loss every step, eval loss every 250 steps. Run ≥ 2 epochs. Save `sft.pt`.
Artifact: `sft.pt` exists; loss curve in `runs/`.

**2.5 Qualitative SFT eval.**
`eval.py` that loads base vs. SFT and generates on 10 held-out prompts side-by-side.
Write observations in `notes/02-sft.md`.
Artifact: side-by-side output pasted into notes.

---

### Module 3 — Reward Model

**3.1 RM architecture.**
`GPT` backbone + `nn.Linear(n_embd, 1)` head. Initialize from `sft.pt` (not base).
Forward returns per-token scalar; reward = value at last non-pad token.
Artifact: shape test passes.

**3.2 Preference dataset.**
Tokenize `prompt + chosen` and `prompt + rejected` separately. Return both with attention
masks and last-token indices.
Artifact: dataset shape/dtype test passes.

**3.3 Bradley–Terry loss.**
`L = -log σ(r_c - r_r)`. Derive ∂L/∂r_c and ∂L/∂r_r in `notes/03-rm.md`, including the
softplus form for numerical stability. Gradient-check at fp64.
Artifact: test_grad_rm.py passes.

**3.4 RM training loop.**
lr=1e-5, 1 epoch over pairs, eval = pairwise accuracy (`r_c > r_r`) on held-out.
Target: ≥ 65% pairwise accuracy. Save `rm.pt`.
Artifact: `rm.pt` exists; accuracy logged.

**3.5 Reward calibration.**
Plot histograms of `r_c` and `r_r` on eval. They should overlap but `mean(r_c) > mean(r_r)`.
Artifact: histogram saved; note in `notes/03-rm.md`.

---

### Module 4 — PPO building blocks

Each function lives in `ppo_core.py` with its own gradient test.

**4.1 Rollout: `generate_with_logprobs(policy, prompts, max_new_tokens)`.**
Returns `response_tokens`, `logprobs_old`, `values_old`, `attention_mask`, `response_mask`.
Single forward per step. Critical: log-prob at position $t$ must be
$\log p(\mathrm{token}_{t+1} \mid \mathrm{prefix}_{\le t})$.
Artifact: shape/alignment test passes.

**4.2 Per-token KL.**
`kl_k1 = logprobs - ref_logprobs` (shape `[B, T_resp]`).
Include `kl_k3 = (ratio - 1) - log(ratio)` as the logging estimator.
Explain the variance tradeoff in `notes/04-ppo-kl.md`.
Artifact: unit test that k1 integrates to expected KL on a known distribution.

**4.3 Reward shaping.**
Terminal reward $r_\text{RM}$ from RM at `<|im_end|>`; per-token reward
$r_t = -\beta \cdot \mathrm{KL}_t + r_\text{RM} \cdot \mathbf{1}_{t=T}$.
Test: $\beta \to 0$ collapses to pure RM terminal reward.
Artifact: unit test passes.

**4.4 GAE.**
Given $r_t, V_t, V_{t+1}$ and a done-mask, compute
$A_t = \sum_k (\gamma \lambda)^k \delta_{t+k}$ where
$\delta_t = r_t + \gamma V_{t+1} - V_t$.
Returns $= A + V$ (value target). Unit-test against a hand-computed 3-step example.
Derive in `notes/04-ppo-gae.md`.
Artifact: test passes with hand-computed example.

**4.5 PPO clipped policy loss.**

$$
\mathcal{L}_\pi = -\mathbb{E}\!\left[ \min\!\left( \rho A, \; \mathrm{clip}(\rho, 1-\varepsilon, 1+\varepsilon) \cdot A \right) \cdot \mathrm{mask} \right]
$$

Derive piecewise gradient in `notes/04-ppo-policy.md`. Gradient-check at fp64.
Include edge test where every ratio is clipped — gradient must be zero on those tokens.
Artifact: test_grad_ppo.py::test_ppo_policy_loss_grad passes.

**4.6 Value loss (clipped).**

$$
\mathcal{L}_V = \tfrac{1}{2} \, \max\!\left( (V - R)^2, \; (\mathrm{clip}(V, V_\text{old} - \varepsilon_v, V_\text{old} + \varepsilon_v) - R)^2 \right) \cdot \mathrm{mask}
$$

Gradient-check. Explain why clipping value helps early training in `notes/04-ppo-policy.md`.
Artifact: test_grad_ppo.py::test_value_loss_grad passes.

**4.7 Entropy bonus.**
Mean of $-\sum_a \pi(a) \log \pi(a)$ over masked response tokens. Start at coefficient 0.
Grad-check.
Artifact: test_grad_ppo.py::test_entropy_grad passes.

**4.8 Advantage normalization.**
Per-batch normalize advantages (subtract mean, divide std) over valid tokens only.
Unit test: pad half the batch with garbage; verify stats are unchanged.
Artifact: test passes with padded batch.

---

### Module 5 — PPO training loop

**5.1 Model layout and memory map.**
Instantiate policy, value head (shared backbone, separate head), frozen ref, frozen RM.
Print param counts and expected bf16 memory. `torch.cuda.max_memory_allocated` after one
dummy forward must fit within 24GB.
Artifact: memory printout in `notes/05-ppo.md`.

**5.2 Outer loop (rollout phase).**
For each iteration: sample a batch of prompts, generate responses, compute ref logprobs
(no grad), compute RM reward (no grad), run GAE, store in a dict of tensors.
Artifact: one rollout iteration runs without error.

**5.3 Inner loop (optimize phase).**
For $K = 4$ epochs, iterate minibatches, compute
$\mathcal{L} = \mathcal{L}_\pi + c_v \mathcal{L}_V - c_\text{ent} H$, backward, clip
grad norm to 1.0, step. Recompute $\log \pi$ and $V$ freshly each minibatch;
$\log \pi^\text{old}$ is frozen from rollout.
Artifact: one full inner loop runs without error.

**5.4 Logging.**
Per-iter: mean reward, mean KL (k3), policy loss, value loss, entropy, clip fraction, grad
norm, tokens/sec. Save CSV; emit matplotlib plots every 50 iters.
Expected trends: KL up slowly, reward up, entropy down slowly.
Reward up while KL explodes = reward hacking → lower LR or raise β.
Artifact: CSV and plots generated after 10 iters.

**5.5 Config scaling.**
`GPTConfig.from_name("gpt2-small"|"gpt2-medium"|"gpt2-large"|"gpt2-xl")`.
Smoke test: one forward+backward at batch=1, seq=64 for each size.
small/medium must step; large/xl may OOM.
Artifact: memory table in `notes/05-ppo.md`.

---

### Module 6 — Evaluation

**6.1 Generation comparison.**
`eval.py` emits a markdown table of 20 held-out prompts × 3 models (base, SFT, RLHF).
Artifact: `notes/06-eval.md` with the full table.

**6.2 Win-rate (manual).**
Self-annotate 20 SFT-vs-RLHF pairs blinded. Target: RLHF wins ≥ 55%.
If not, ablate: lower β, more PPO epochs, longer RM training.
Artifact: win-rate tally in `notes/06-eval.md`.

**6.3 Retrospective.**
What was hard, what you'd do differently, which grad check saved you, where the old code
was wrong.
Artifact: `notes/06-eval.md` retrospective section.

---

## 7. Hyperparameters

**SFT:** lr=3e-5, bs=64 (via accum), epochs=2, wd=0.1, warmup=200, cosine→10%.

**RM:** lr=1e-5, bs=32, epochs=1, init from `sft.pt`.

**PPO:** lr=1e-6 (policy) / 1e-5 (value head), rollout batch=64 prompts, response_len=128–256,
K=4, minibatch=16, γ=1.0 (episodic), λ=0.95, ε=0.2, ε_v=0.2, β=0.02, c_ent=0.0.

---

## 8. Data: Anthropic HH-RLHF

Source: `Anthropic/hh-rlhf`. Raw format: `{"chosen": str, "rejected": str}` — multi-turn
dialogues alternating `Human:` / `Assistant:`.

Chat template:
```
<|im_start|>user
<turn text><|im_end|>
<|im_start|>assistant
<turn text><|im_end|>
```

Three derived datasets from `data_hh.py`:

1. **SFT:** full chosen dialogue. `loss_mask` = 1 on assistant tokens only (including `<|im_end|>`).
2. **Preference pairs:** `(prompt, chosen_response, rejected_response)` for the RM.
3. **Prompt-only:** prompts ≤ 512 tok; rollouts ≤ 256 new tokens.

`<|im_start|>` and `<|im_end|>` are not in base GPT-2 BPE — encoded as literal UTF-8 bytes.

---

## 9. Hardware & memory budget (24GB GPU)

GPT-2 small = 124M params. PPO holds four models simultaneously:

| Model | Params | Grads | AdamW state |
|---|---|---|---|
| Policy (trainable) | 124M | yes | fp32 m, v |
| Value backbone + head (trainable, shared) | 124M | yes | fp32 m, v |
| Reference policy (frozen) | 124M | no | — |
| Reward model (frozen) | 124M | no | — |

Rough bf16 budget with gradient checkpointing:
- Weights: 4 × 124M × 2B ≈ 1.0 GB
- fp32 master + Adam state: ≈ 1.5 GB
- Activations (batch 4 × seq 512): 4–8 GB
- Rollout buffers: < 1 GB

---

## 10. Gradient-check protocol

Every loss gets a dedicated test. Template:

```python
def test_<loss>_grad():
    torch.manual_seed(0)
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
- fp64 + tiny dims (e.g. 2 layers, n_embd=16, 1–2 heads, seq 8, batch 2).
- Check gradients w.r.t. both inputs and parameters.
- **Mask test:** flip a masked-out element and assert the loss is unchanged.

---

## 11. Debugging tips

- **Grad checks lie only when you lie to them.** If a check passes but training does
  nothing, the bug is in masking / shift / reduction — check `.shape` and `.sum()` of
  every mask.
- **PPO can appear to train while silently broken.** Watch all six logs simultaneously:
  reward, KL (k3), policy loss, value loss, entropy, clip fraction.
- **Alignment bugs in PPO** (logits[t] vs response[t]): write a 3-token example on paper
  and verify indices before touching code.
- **If your results don't match expectations**, re-read the gradient derivations in the
  corresponding `notes/` file and verify every mask and index in the backward pass.

---

## 12. Completion criteria

- [ ] `python -m pytest tests/ -q` is fully green.
- [ ] `python eval.py --models base,sft,rlhf --n 20` produces a table where RLHF is
      visibly better on most rows.
- [ ] You can derive, on a whiteboard, the backward pass of every loss in the repo.
- [ ] `notes/` has one markdown per module with your own derivations.
- [ ] `gpt2-small` trains end-to-end on a 24GB GPU via `config.py`.
- [ ] No dependency on `transformers`, `trl`, `accelerate`, `peft`, or `deepspeed`.

---

## 13. Norms

- No `transformers.Trainer`, no `trl`, no `accelerate`, no `peft`, no `deepspeed`.
  `transformers` is only allowed in `load_gpt2_from_hf` for the one-time weight load.
- No `ignore_index=-100`. Always an explicit `loss_mask` multiplied in.
- `nn.Module` yes; dataclasses yes; anything more abstract — no.
- Comments explain *why*, not *what*.