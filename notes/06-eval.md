# 06 — Evaluation

## Purpose

Methodology for Module 6 (Problems 6.1 through 6.3). There is almost no math left
at this stage — the theory is all behind you. This note is about:

1. How to fairly compare three models (base vs SFT vs RLHF).
2. How to compute a win-rate from blinded pairwise annotations.
3. What to write in your retrospective.

Fill this file in with your actual outputs as you run the problems.

Evaluation is where you stop optimizing proxies and look at behavior. Training loss, reward
model score, KL, and win-rate are all useful, but none of them substitutes for reading the
model's answers. The final question is not "did a scalar improve?" It is "does this assistant
follow instructions better in ways a human can recognize?"

Read evaluation by looking at behavior, not only metrics. Base GPT-2 should reveal why
instruction tuning is needed. SFT should show what imitation buys you. RLHF should show
whether optimizing the learned preference signal improved answers beyond imitation or mostly
changed their style. The side-by-side table is where those differences become visible.

---

## 1. Generation comparison (Problem 6.1)

Build a table with one row per held-out prompt (target: 20 prompts) and one
column per model (base, SFT, RLHF). Each cell holds the generated response,
trimmed to the first `<|im_end|>` or to a reasonable cutoff if the model never
stops.

Keep the prompts fixed and save them. A stable prompt set lets you compare runs across
hyperparameter changes. If every run uses a new random set of prompts, you will never know
whether a difference came from the model or the sample.

### 1.1 Prompt selection

Pick 20 prompts from the HH test split that were **not** used in SFT or PPO
training. A reasonable mix:

- 5 helpful requests ("write me a Python function that ...").
- 5 conversational openers ("I've been feeling stressed about ...").
- 5 factual questions ("What's the capital of Kenya?").
- 5 edge cases where base GPT-2 is known to fail: long multi-step instructions,
  requests for specific formats (like JSON or tables), requests for lists.

The mix matters because models can improve unevenly. SFT may mostly improve role-following.
RLHF may mostly improve helpfulness or refusal style. A prompt set that contains only easy
factual questions will miss formatting, conversational, and instruction-following failures.

### 1.2 Generation settings

**Keep them identical across all three models.** Temperature 1.0, top-p 0.9 (or
whatever you settle on). Use the same random seed per prompt across models. Any
difference you see should be attributable to the model, not to sampling
variance.

For a more deterministic comparison you can also generate with temperature 0
(greedy). More boring to read, but removes all sampling noise so the differences
are fully attributable to the model weights.

A good practice is to run both: greedy for a clean model-to-model comparison, and sampled
generation for a more realistic view of behavior. Greedy outputs can hide diversity problems;
sampled outputs can hide regressions behind randomness. Together they give a better picture.

### 1.3 Format

A markdown table like:

```
| # | Prompt | Base | SFT | RLHF |
|---|--------|------|-----|------|
| 1 | ...    | ...  | ... | ...  |
```

Paste it under a "Results" section in this file.

When the table is large, readability matters. Trim extremely long responses, but do not trim
away the failure. If a model rambles, show enough rambling that the failure mode is visible.
If a model never emits `<|im_end|>`, note that explicitly.

---

## 2. Win-rate (Problem 6.2)

### 2.1 Setup

On the same 20 prompts, compare the SFT and RLHF responses pairwise. For each
prompt:

- Present both responses to yourself *without knowing which is which*. Write a
  simple randomizer that swaps the order with 50% probability and keeps the
  answer hidden.
- Rate one as "better" — more helpful, more accurate, more aligned with the
  instruction — or declare a tie.

### 2.2 Guarding against position bias

Humans (and LLM-as-judge evaluators) have a systematic preference for whichever
option they see first. Two mitigations:

- Randomize the order (you already did this above).
- Run the 20 comparisons once, then re-run them with order reversed, and average
  the results. If your ratings agree across the two passes, the randomizer was
  enough; if they differ, averaging corrects the bias.

### 2.3 Scoring

    win_rate_RLHF = (RLHF_wins + 0.5 * ties) / 20

Target: at least 0.55 — RLHF is meaningfully better than SFT on a majority of
prompts. If the win-rate is 0.5 or below, RLHF is no better than (or worse than)
SFT and the run failed. Likely reasons:

- Weak reward model.
- `beta` too large — the policy barely moved from SFT.
- Too few PPO iterations.
- Reward hacking that isn't visible on these particular prompts.

Treat 20 prompts as a smoke test, not a publication-grade benchmark. A 55% win-rate on 20
items is only a weak signal, but it is enough to catch obvious regressions and to force
manual inspection. If you want stronger evidence, increase the prompt count and keep the
annotation protocol blinded.

### 2.4 If RLHF didn't win

Don't paper over it. Write down honestly:

- What the RLHF responses look like that SFT's don't.
- What the RLHF responses are missing that you'd want.
- Which hyperparameter you'd change first, and why.

Then actually change it and try again, if you want to.

The most useful failed evaluation is specific. "RLHF lost" is not specific. "RLHF gives
longer answers that score well but ignore requested JSON formatting" is specific, and it
points toward checking reward length bias and format-sensitive prompts.

---

## 3. The retrospective (Problem 6.3)

One page, answering these six questions directly.

The retrospective should be technical, not sentimental. Name the bug, the symptom, the test
that caught it, and the fix. The point is to make the next RLHF implementation easier because
you have a written catalog of failure modes.

### 3.1 What was the hardest part?

Some candidates from past RLHF projects: aligning `logprobs_old` and
`logprobs_new` in the PPO loop; deciding between shared backbone and separate
value network; getting RM pairwise accuracy above the noise floor; managing
memory with four models loaded at once.

### 3.2 Where did you actually spend the most time?

Count your commits and your notes. The gap between "how hard I thought it would
be" and "how long it actually took" is diagnostic. Almost always, the answer
reveals where your mental model was the weakest.

This is worth writing down because time spent is a better teacher than confidence. If a
concept felt easy but consumed two days, that concept deserves another derivation or a better
unit test.

### 3.3 Which gradient check saved you?

For every test in `tests/`, ask yourself: did it catch a real bug? Which bug?
Which tests were redundant? Use this to decide what to keep testing in your next
project.

### 3.4 What would you do differently next time?

Candidates: separate value network; adaptive `beta`; sample more prompts during
rollout; use `k_3` instead of `k_1` for the KL penalty; explicitly handle length
bias in the RM; initialize the value head from SFT instead of zero.

### 3.5 What did you get wrong and fix?

Trace back through your git history and identify bugs you caught during
development:

- Did you ever mask the wrong tokens in SFT? (Included user turns? Forgot to
  include `<|im_end|>`? Used an inverted mask?)
- Did you have an off-by-one or alignment bug between logprobs and tokens in the
  PPO rollout?
- Did you forget to clip the value loss, or compute advantages without masking?

These are the educational points of the whole exercise — the real bugs that
appear when you build RLHF from scratch, and how you learned to catch them.

### 3.6 What did you not understand before, and what do you understand now?

Pick one thing. Write two paragraphs. "I used to think X; now I think Y." If you
can't think of one, you didn't learn anything and you should redo the course.
But you *will* find one.

Good examples are concrete: "I used to think PPO clipping just limited ratios; now I
understand it clips only advantage-improving moves." Or: "I used to think masking was a
data-loader detail; now I understand it defines the actual supervised objective."

---

## 4. What to commit to `notes/06-eval.md`

- The 20-row generation table from 6.1.
- The 20-row win-rate tally from 6.2 with your scoring notes.
- The retrospective from 6.3.

This is the last file in the curriculum. When it's complete, the course is
complete.

At that point, the notes should be useful without the code open. A future reader should be
able to understand what was trained, how it was evaluated, which results were convincing, and
which parts still need work.
