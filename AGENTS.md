# LearnRLHF — InstructGPT PPO from scratch on GPT-2 (crash-course edition)

## 0. Mission

A teaching-grade implementation of InstructGPT-style RLHF (Ouyang et al. 2022) on
GPT-2 small, trainable on a single 24GB RTX GPU, using the Anthropic HH-RLHF dataset.

Format: **fill-in-the-blank, CS231n style, sized for one focused weekend.** All
plumbing (data pipeline, training loops, rollout generation, weight loading, eval)
is provided and tested. The learner (`daniel@`) writes only the parts that carry the
ideas — about 14 blanks, each a few lines, each with a formula in its docstring and
a dedicated test. Reference implementations live in `solutions/` and the test suite
runs against them with `LEARNRLHF_SOLUTIONS=1 pytest tests/`.

The learner must come out able to:

1. Derive every loss gradient used across SFT, RM, and PPO on paper.
2. Explain why each loss has the form it does and what each term buys.
3. Train a GPT-2 small that is qualitatively better at instruction following than
   the pretrained checkpoint, and know which knobs matter.

Style target: Karpathy `nanoGPT` + CS231n assignments. Flat `.py` files, minimal
abstractions, no frameworks, aggressive gradient checking, prose-heavy docstrings.

## 1. Scope & non-goals

In scope:
- Pure PyTorch. Deps: `torch`, `tiktoken`, `datasets`, `numpy`, `matplotlib`,
  `tqdm`. `transformers` optional, used only to download GPT-2 weights.
- GPT-2 re-implemented: tied embeddings, learned positional embeddings, LayerNorm
  with affine params, exact GELU.
- SFT, Reward Model, PPO — all three InstructGPT phases.
- bf16 autocast, gradient accumulation, optional gradient checkpointing.
- Single GPU. `GPTConfig.from_name` instantiates small/medium/large/xl even if the
  big ones OOM in training.

Out of scope: DDP/FSDP/ZeRO/DeepSpeed, LoRA/PEFT, custom kernels, KV cache (unless
rollouts are unbearably slow), tokenizer training, labeling UI, LLM-as-judge eval.

## 2. The course (weekend schedule)

Each blank is marked in the source as `[FILL N.M]` with a `YOUR CODE` banner. Tests
skip while a blank is unfilled and fail only on wrong implementations.

| Part | When | Blanks | Files |
|---|---|---|---|
| 1. Model | Sat AM (~3h) | 1.1 attention math, 1.2 GPT.forward_hidden | `model.py` |
| 2. SFT | Sat PM (~2h + GPU) | 2.1 build_sft_example (loss mask), 2.2 sft_loss | `tokenizer.py`, `train_sft.py` |
| 3. RM | Sat eve (~1.5h + GPU) | 3.1 last-token pooling, 3.2 bt_loss | `train_rm.py` |
| 4. PPO core | Sun (~4-5h) | 4.1 gather_logprobs, 4.2 kl_k1/kl_k3, 4.3 shape_reward, 4.4 gae, 4.5 ppo_policy_loss, 4.6 value_loss, 4.7 masked_entropy, 4.8 normalize_advantages | `ppo_core.py` |
| 5. Run + eval | Sun eve | none — run `train_ppo.py`, then `eval.py` | provided |

Gate between parts: the relevant tests must be green before moving on. The smoke
tests (`tests/test_training_smoke.py`) are the gate for Part 4 — they train tiny
models on CPU and assert SFT loss drops, RM accuracy rises, and PPO reward climbs.

## 3. Hardware & memory budget (24GB)

GPT-2 small = 124M params. PPO holds four models:

| Model | Trainable | Notes |
|---|---|---|
| Policy | yes | init from sft.pt |
| Value head | yes | ScalarHead on the policy's hidden states (shared backbone) |
| Reference | no | frozen sft.pt, the KL anchor |
| Reward model | no | frozen rm.pt |

bf16 weights for 4 backbones ~1GB; fp32 Adam state for the trainable ones ~1.5GB;
activations for batch 8 x seq 640 a few GB; rollout buffers <1GB. Comfortable on
24GB. `gradient_checkpointing=True` in GPTConfig buys headroom for gpt2-medium.

All hyperparameters live in `config.py`; model size switches with
`GPTConfig.from_name`.

## 4. Repo layout

```
config.py            # GPTConfig (+from_name), SFTConfig, RMConfig, PPOConfig
model.py             # GPT-2, HF weight load, sampling          [blanks 1.1, 1.2]
tokenizer.py         # tiktoken + chat template + loss mask     [blank 2.1]
data_hh.py           # HH download, 3 derived datasets          (provided)
train_sft.py         # sft_loss + training loop                 [blank 2.2]
train_rm.py          # RewardModel + bt_loss + loop             [blanks 3.1, 3.2]
ppo_core.py          # 9 PPO building blocks + rollout          [blanks 4.1-4.8]
train_ppo.py         # rollout/optimize loop, CSV logs, plots   (provided)
eval.py              # base vs SFT vs RLHF markdown table       (provided)
grad_check.py        # fp64 centered-difference checker         (provided)
solutions/           # complete reference implementations
tests/               # one test per blank + smoke training tests
notes/               # theory references, one per part
```

## 5. Data: Anthropic HH-RLHF

`Anthropic/hh-rlhf` from HuggingFace. Raw rows: `{"chosen": str, "rejected": str}`,
multi-turn `Human:` / `Assistant:` dialogues differing in the final assistant turn.

Chat template, enforced end-to-end:

```
<|im_start|>user
<turn text><|im_end|>
<|im_start|>assistant
<turn text><|im_end|>
```

`<|im_start|>`/`<|im_end|>` are encoded as literal UTF-8 bytes (no vocab surgery).
The one real special token is `<|endoftext|>` (50256): SFT appends it after the
final assistant turn (loss mask 1) so the model learns to stop; generation and PPO
use it as EOS.

Three derived datasets in `data_hh.py`:
1. **SFT**: chosen dialogue, `loss_mask` = 1 only on assistant tokens. Right-padded.
2. **Preference pairs**: chosen and rejected tokenized independently, plus last-real-
   token indices for reward pooling. Right-padded.
3. **Prompt-only**: dialogue minus final assistant turn, with the open
   `<|im_start|>assistant\n` cue, prompts <= 512 tokens. LEFT-padded for rollouts.

Padding contract (load-bearing, see `test_gpt_left_padding_consistent`): attention
masks hide pad keys with a large finite negative (not -inf, which NaNs fully-masked
rows), and position ids are derived from the attention mask, so left-padded rows
produce identical logits to their unpadded versions.

## 6. Gradient-check protocol

Every loss gets a centered-difference check at fp64 with tiny dims via
`grad_check.check_grad`. Rules:

- fp64 + tiny shapes (n_embd 16, heads 2, seq 8, batch 2). fp32 checks chase noise.
- At least one mask test per masked loss: flip a masked-out element, assert the loss
  is bit-identical.
- Edge tests where the loss is piecewise: e.g. all-ratios-clipped must give exactly
  zero gradient in the PPO policy loss.

## 7. Backward-pass mental model (derive these on paper before filling each blank)

1. **SFT.** L = -(1/N) Σ_t m_t log softmax(z_t)[y_t], N = Σ m_t.
   dL/dz_t = m_t (softmax(z_t) - onehot(y_t)) / N.

2. **Reward model (Bradley–Terry).** L = -log σ(r_c - r_r) = softplus(r_r - r_c).
   dL/dr_c = σ(r_c - r_r) - 1 (negative: pushes r_c up); dL/dr_r = 1 - σ(r_c - r_r)
   (positive: pushes r_r down). Exact negatives; only score differences matter.

3. **PPO clipped surrogate.** ratio_t = exp(logπ_t - logπ_old_t),
   L_t = -min(ratio_t · A_t, clip(ratio_t, 1-ε, 1+ε) · A_t). Where the min picks the
   clipped branch, the gradient through logπ_t is exactly zero; elsewhere it is
   -A_t · ratio_t / N. The sign flips with sign(A_t) — that is the policy gradient.

4. **KL penalty.** k1_t = logπ_t - logπ_ref_t: unbiased, signed, the InstructGPT
   shaping penalty. Token reward: r_t = r_RM · 1{t = last} - β · k1_t.
   k3_t = exp(-(logπ_t - logπ_ref_t)) - 1 + (logπ_t - logπ_ref_t): nonnegative
   (since e^x >= 1 + x), lower variance, used for logging only. Note the inverse
   ratio: estimating KL(π‖π_ref) from samples of π uses π_ref/π, not π/π_ref.

5. **Value loss.** L_V = ½ max((V - R)², (clip(V, V_old ± ε_v) - R)²). The target
   R = A + V_rollout is a constant (computed under no_grad) even though it was
   derived from V itself.

6. **Entropy bonus.** H = -Σ_v p_v log p_v; dH/dz = -p ⊙ (log p + H). Added with a
   small (often zero) coefficient against premature determinism.

## 8. Default hyperparameters

SFT: lr 3e-5, effective batch 64 via accumulation, 2 epochs, wd 0.1, warmup 200,
cosine to 10%.

RM: lr 1e-5, batch 32 pairs via accumulation, 1 epoch, init from sft.pt. Target
>= 65% held-out pairwise accuracy.

PPO: policy lr 1e-6, value-head lr 1e-5, rollout batch 32 prompts, response <= 128
tokens, K=4 epochs, minibatch 8, γ=1.0, λ=0.95, clip 0.2, value clip 0.2,
β_KL=0.02, entropy 0.0. Rollouts at temperature 1.0, no top-k/top-p — the recorded
log-probs must describe the distribution tokens were actually drawn from.

Healthy PPO logs: reward up, KL(k3) up slowly, entropy down slowly, clip fraction
0.05–0.3. Reward spiking while KL explodes = reward hacking → raise β or lower lr.

## 9. Definition of done

- [ ] `pytest tests/` green with no skips except the slow-marked HF parity test.
- [ ] `LEARNRLHF_SOLUTIONS=1 pytest tests/` green (solutions stay verified).
- [ ] `python eval.py --models base,sft,rlhf --n 20` table where the RLHF column is
      visibly better at instruction following on a majority of prompts.
- [ ] Learner can derive on a whiteboard: SFT grad, BT grad, PPO clipped-ratio grad,
      GAE recursion, k1 vs k3, value clip, entropy gradient.
- [ ] `gpt2-small` trains end-to-end on one 24GB GPU.
- [ ] No dependency on `trl`, `accelerate`, `peft`, or `deepspeed`
      (`transformers` only for the weight download).

## 10. Working norms for Codex inside this repo

- **The blanks are the course.** Never fill a `[FILL]` blank in the scaffold files
  for the learner. Solutions changes go in `solutions/` and must keep
  `LEARNRLHF_SOLUTIONS=1 pytest tests/` green.
- **One loss, one blank, one test.** Any new loss lands as scaffold blank +
  solutions implementation + test in the same change.
- **Never `ignore_index=-100`**; always an explicit mask tensor multiplied in.
  Masking bugs are the whole point of this course.
- **No new abstractions.** `nn.Module` and dataclasses, nothing fancier.
- **No emojis in code or notes.**
- **Prefer `torch.testing.assert_close` with explicit tolerances**, except inside
  grad-check code where the relative error should be visible.
- **If you can't state a gradient in closed form, stop and derive it** before
  writing the code.
