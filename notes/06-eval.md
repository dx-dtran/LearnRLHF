# 06 — Evaluation

## Purpose

Methodology for Module 6 (Problems 6.1 through 6.3). There is very little math left at
this stage — the theory is all behind you. This note is about:

1. How to compare three models fairly (base vs SFT vs RLHF).
2. How to compute a win-rate from blinded pairwise annotations.
3. What to write in the retrospective.

Fill this file in with your actual outputs as you run the problems.

---

## 1. Generation comparison (Problem 6.1)

Build a table with one row per held-out prompt (target: 20 prompts) and one column per
model (base, SFT, RLHF). Each cell contains the generated response, trimmed to the
first `<|im_end|>` or a reasonable cutoff.

### 1.1 Prompt selection

Pick 20 prompts from the HH test split that were **not** seen in SFT or PPO training.
A good mix:

- 5 helpful requests ("write me a Python function that ...").
- 5 conversational turns ("I've been feeling stressed about ...").
- 5 factual questions ("What's the capital of Kenya?").
- 5 edge cases where base GPT-2 is known to fail (long multi-step instructions,
  requests for specific formats, requests for lists).

### 1.2 Generation settings

**Identical across all three models.** Temperature $1.0$, top-$p$ $0.9$ (or whatever
you end up using). Same random seed per prompt across models. Any difference you see
should be attributable to the model, not to sampling.

For deterministic comparison you can also generate with temperature 0 (greedy) — it's
more boring but removes sampling noise.

### 1.3 Format

A markdown table like:

```
| # | Prompt | Base | SFT | RLHF |
|---|--------|------|-----|------|
| 1 | ...    | ...  | ... | ...  |
```

Paste it into this file under a "Results" section.

---

## 2. Win-rate (Problem 6.2)

### 2.1 Setup

On the same 20 prompts, compare SFT vs RLHF responses pairwise. For each prompt:

- Present both responses to yourself *without knowing which is which*. Write a
  randomizer that swaps the order with 50% probability and keeps a hidden record.
- Rate one as "better" — more helpful, more accurate, more aligned with the
  instruction — or declare a tie.

### 2.2 Guarding against position bias

Humans (and LLM judges) systematically prefer the first option they see. Mitigate:

- Randomize order (done above).
- Over-sample and recompute: do the 20 comparisons, then re-do the same 20 with
  reversed order, and average the two. If your judgments are consistent, the
  randomizer is sufficient; otherwise average.

### 2.3 Scoring

$$
\text{win\_rate}_\text{RLHF} = \frac{\#\{\text{RLHF wins}\} + 0.5 \cdot \#\{\text{ties}\}}{20}.
$$

Target: $\ge 0.55$ (RLHF meaningfully better than SFT on a majority of held-out
prompts). At $\le 0.5$, RLHF is no better than (or worse than) SFT — the run failed.
Probable causes: weak reward model, $\beta$ too large so the policy barely moved, too
few PPO iterations, reward hacking that doesn't look like hacking on these specific
prompts.

### 2.4 If RLHF didn't win

Don't paper over it. Write down:

- What the RLHF responses look like that SFT doesn't.
- What the RLHF responses are missing that you'd want.
- Which hyperparameter you'd change next.

Then actually change it and try again if you want to.

---

## 3. The retrospective (Problem 6.3)

One page. Answer these six questions directly:

### 3.1 What was the hardest part?

Candidates from past RLHF projects include: PPO alignment between `logprobs_old`
and `logprobs_new`; getting the shared-backbone / separate-value-head design right;
the RM training and keeping pairwise accuracy above noise; memory management in
PPO with four models.

### 3.2 Where did you spend the most time?

Count your commits or your notes. The gap between how hard you thought it would be
and how long it actually took is diagnostic.

### 3.3 Which gradient check saved you?

For every test in `tests/`, ask: did it catch a real bug? Which bug? Which tests were
redundant? You'll use this to decide what to keep testing in the next project.

### 3.4 What would you do differently next time?

Possible answers: separate value network; adaptive $\beta$; sample more prompts
during rollout; use $k_3$ instead of $k_1$ for the KL penalty; handle length bias
in the RM explicitly; start value head from SFT instead of zero.

### 3.5 What did you get wrong and fix?

Trace back through your git history and identify bugs you caught during development:
- Did you ever mask the wrong tokens in SFT? (e.g., included user turns, excluded
  `<|im_end|>`, or used inverted mask).
- Did you have an off-by-one or alignment bug in the PPO rollout (logprobs vs tokens)?
- Did you forget to clip the value loss, or compute advantages without masking?

These are the educational points of the exercise — real bugs that arise when implementing
RLHF from scratch, and how you caught them.

### 3.6 What did you not understand before, and what do you understand now?

Pick one thing. Write two paragraphs. "I used to think $X$; now I think $Y$." If
you can't think of one, you didn't learn anything and you should do the course
again. You *will* find one.

---

## 4. What to commit to `notes/06-eval.md`

- The 20-row generation table from 6.1.
- The 20-row win-rate tally from 6.2 with your scoring notes.
- The retrospective from 6.3.

This is the last file in the curriculum. When it's complete, the course is complete.
