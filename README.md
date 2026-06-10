# LearnRLHF — InstructGPT PPO on GPT-2, as a weekend crash course

From-scratch InstructGPT-style RLHF (Ouyang et al. 2022) on GPT-2 small, single 24GB
GPU, pure PyTorch. No `trl`, no `accelerate`, no `transformers.Trainer`.

This is a fill-in-the-blank course in the CS231n style. All the plumbing — data
pipeline, training loops, rollout generation, weight loading, evaluation — is written
and tested. You write the parts that carry the ideas: the losses, the advantage
estimator, the KL penalty, the attention math. Each blank is a handful of lines, has
the formula in its docstring, and has a test (usually a gradient check) waiting for
it. Roughly 14 blanks, 12–16 focused hours total.

The goal hasn't changed: at the end you can derive every gradient in SFT, RM, and PPO
on paper, and you have a GPT-2 that visibly follows instructions better than the
pretrained checkpoint.

## How it works

Every blank looks like this:

```python
    # ================================ YOUR CODE (~5 lines) ==========================
    raise NotImplementedError("[FILL 4.5] ppo_policy_loss")
```

The docstring above it states the math. Replace the `raise` with your implementation
and run the tests:

```bash
pytest tests/
```

Unfilled blanks show as **skipped**. A **failed** test always means
implemented-but-wrong. Start to finish, the test summary is your progress bar: when
nothing is skipped (except the marked-slow HF parity test) and nothing fails, the
code is done.

Reference implementations live in `solutions/`. Try each blank yourself first —
the course is the blanks — but if you're stuck for more than ~20 minutes, look. You
can also run any test against the solutions to check whether a failure is your code
or your understanding of the test:

```bash
LEARNRLHF_SOLUTIONS=1 pytest tests/
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest tests/test_grad_check.py -q     # 3 passed = environment works
```

## The schedule

### Saturday morning — Part 1: the model (~3 h)

Read `notes/01-gpt2.md` as needed. LayerNorm, the MLP, the block structure, weight
loading, and sampling are provided; you write the two pieces that matter:

| Blank | Where | What |
|---|---|---|
| 1.1 | `model.py` | the attention math: qkv, scale, causal + padding masks, softmax, projection |
| 1.2 | `model.py` | `GPT.forward_hidden`: embeddings, blocks, final LN |

Tests: `pytest tests/test_model.py`. The padding tests are not pedantry — the PPO
phase left-pads prompts, and `test_gpt_left_padding_consistent` is the contract the
whole pipeline relies on. When green, run the (slow, downloads ~500MB) parity check
once: `pytest tests/test_model.py -m slow`.

### Saturday afternoon — Part 2: SFT (~2 h + GPU time)

Read `notes/02-sft.md`. The dataset, collate, optimizer, and training loop are
provided.

| Blank | Where | What |
|---|---|---|
| 2.1 | `tokenizer.py` | `build_sft_example`: tokenize a dialogue with a loss mask that is 1 only on assistant tokens |
| 2.2 | `train_sft.py` | `sft_loss`: masked next-token cross-entropy |

Derive `dL/dlogits` on paper before you code 2.2 — it's the one gradient everyone
memorizes, derive it anyway. Tests: `test_tokenizer.py`, `test_data.py`,
`test_grad_sft.py`. Then kick off training (a few hours on a 24GB card; let it run
while you do Part 3's math):

```bash
python train_sft.py        # writes sft.pt
```

### Saturday evening — Part 3: reward model (~1.5 h + GPU time)

Read `notes/03-rm.md`.

| Blank | Where | What |
|---|---|---|
| 3.1 | `train_rm.py` | `RewardModel.forward`: pool per-token scores at the last real token |
| 3.2 | `train_rm.py` | `bt_loss`: Bradley–Terry, `softplus(r_rejected - r_chosen)` |

Derive both gradients of the pair loss and notice they're exact negatives. Tests:
`test_grad_rm.py`. Then:

```bash
python train_rm.py         # needs sft.pt; writes rm.pt; expect >= 65% pairwise acc
```

### Sunday — Part 4: PPO core (~4–5 h)

The heart of the course. Read `notes/04-ppo-gae.md`, `notes/04-ppo-kl.md`,
`notes/04-ppo-policy.md` alongside the blanks. All nine live in `ppo_core.py`, each
with formula in docstring and its own test in `test_grad_ppo.py`:

| Blank | What | The thing to understand |
|---|---|---|
| 4.1 | `gather_logprobs` | logits at position t score token t+1 — the caller slices `[:, T_p-1:-1]` |
| 4.2 | `kl_k1`, `kl_k3` | unbiased-but-signed penalty vs nonnegative logging estimator |
| 4.3 | `shape_reward` | RM reward lands on the LAST real token; KL penalty on every token |
| 4.4 | `gae` | the backward recursion; pad is terminal |
| 4.5 | `ppo_policy_loss` | clipped tokens have exactly zero gradient — prove it, then test it |
| 4.6 | `value_loss` | clipped regression; returns are a constant (stop-grad) |
| 4.7 | `masked_entropy` | dH/dlogits = -p (log p + H) |
| 4.8 | `normalize_advantages` | masked mean/std — THE classic PPO bug |

The rollout function `generate_with_logprobs` is provided; read it line by line, the
off-by-one it handles is the same one you must respect in 4.1.

When `test_grad_ppo.py` is green, run the smoke tests — they train tiny models on
CPU and assert SFT loss drops, RM accuracy rises, and PPO reward climbs:

```bash
pytest tests/test_training_smoke.py -q
```

If those pass, your math is not just plausibly right, it demonstrably optimizes.

### Sunday evening — Part 5: the real run (~30 min of your time + GPU hours)

`train_ppo.py` is fully provided — read it top to bottom (it's the glue you just
built parts for), then:

```bash
python train_ppo.py        # needs sft.pt + rm.pt; writes rlhf.pt + ppo_log.csv
```

Healthy training: mean reward up, KL (k3) drifting up slowly, entropy down slowly,
clip fraction roughly 0.05–0.3. Reward spiking while KL explodes is reward hacking —
raise `kl_coef` or lower `policy_lr` in `config.py`. Plots land in `ppo_plots.png`
every 50 iters.

Finally, judge it:

```bash
python eval.py --models base,sft,rlhf --n 20 --out notes/06-eval.md
```

Blind yourself to which column is which and score SFT vs RLHF on the 20 prompts.
Target: RLHF wins a majority.

## Repo layout

```
config.py            # all hyperparameters; GPTConfig.from_name() switches model size
model.py             # GPT-2, weight loading, sampling          [blanks 1.1, 1.2]
tokenizer.py         # tiktoken + chat template + loss mask     [blank 2.1]
data_hh.py           # HH-RLHF download + 3 datasets            (provided)
train_sft.py         # SFT loss + training loop                 [blank 2.2]
train_rm.py          # reward model + BT loss + loop            [blanks 3.1, 3.2]
ppo_core.py          # the 9 PPO building blocks                [blanks 4.1–4.8]
train_ppo.py         # rollout/optimize loop, logging, plots    (provided)
eval.py              # side-by-side generation table            (provided)
grad_check.py        # fp64 centered-difference gradient checker
solutions/           # reference implementations of every blank
tests/               # one test (usually a grad check) per blank + smoke training
notes/               # theory + derivations, one file per part
```

## Rules of the house

- Pure PyTorch. Deps: `torch`, `tiktoken`, `datasets`, `numpy`, `matplotlib`,
  `tqdm`. `transformers` is optional and used only to download GPT-2 weights.
- No `ignore_index=-100` tricks, ever. Masks are explicit tensors you multiply in.
  Masking bugs are the whole point of this course.
- Every loss gets a gradient check in fp64 against centered differences. PPO will
  happily "train" for thousands of steps with a sign error; the grad checks are what
  stand between you and that.
- Before coding a loss, write its gradient on paper. The notes show every derivation,
  but do it yourself first.

## Hardware

GPT-2 small (124M) everywhere. PPO holds four models — policy (trainable), value
head (trainable, shares the policy backbone), frozen reference, frozen reward
model — and fits in 24GB with bf16 and room to spare. `GPTConfig.from_name`
instantiates medium/large/xl too; medium roughly fits, large/xl are
"compiles and steps" territory.
