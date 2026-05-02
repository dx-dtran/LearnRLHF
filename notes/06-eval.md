# 06 — Evaluation

## Purpose

Methodology for Module 6 (Problems 6.1 through 6.3). The math from previous
modules does not appear here. This note covers:

1. How to compare three models (base vs SFT vs RLHF) under matched conditions.
2. How to compute a win-rate from blinded pairwise annotations.
3. What to record in the retrospective.

Fill this file in with actual outputs while running the problems.

The evaluation step replaces optimization proxies (training loss, reward, KL,
win-rate proxies) with direct reading of model output. Those proxies are
useful, but the final question is whether the assistant follows instructions
better than the previous version.

Base GPT-2 illustrates why instruction tuning is needed at all. SFT shows what
imitation buys. RLHF shows whether optimizing the learned preference signal
improved answers beyond imitation, or whether it mostly changed their style.
The side-by-side table is where those differences become visible.

---

## 1. Generation comparison (Problem 6.1)

Build a table with one row per held-out prompt (target: 20 prompts) and one
column per model (base, SFT, RLHF). Each cell holds the generated response,
trimmed to the first `<|im_end|>` or to a reasonable cutoff if the model never
stops.

Keep the prompts fixed and save them. A stable prompt set lets runs be
compared across hyperparameter changes. With a new random sample each run,
differences cannot be attributed to the model rather than to the prompt
choice.

### 1.1 Prompt selection

Pick 20 prompts from the HH test split that were **not** used in SFT or PPO
training. A reasonable mix:

- 5 helpful requests ("write me a Python function that ...").
- 5 conversational openers ("I've been feeling stressed about ...").
- 5 factual questions ("What's the capital of Kenya?").
- 5 edge cases where base GPT-2 is known to fail: long multi-step
  instructions, requests for specific formats (such as JSON or tables), and
  requests for lists.

Models can improve unevenly. SFT may mostly improve role-following. RLHF may
mostly improve helpfulness or refusal style. A prompt set restricted to easy
factual questions misses formatting, conversational, and instruction-following
failures.

### 1.2 Generation settings

**Use identical settings across all three models.** Temperature 1.0, top-p 0.9
(or whatever value the rest of the course uses). Use the same random seed per
prompt across models. Differences in output should be attributable to the
model, not to sampling variance.

For a more deterministic comparison, also generate with temperature 0
(greedy). Greedy output is less natural to read but eliminates sampling
noise.

Running both is informative. Greedy gives a clean model-to-model comparison;
sampled generation gives a more realistic view of behavior. Greedy outputs
can hide diversity problems, and sampled outputs can hide regressions behind
randomness.

### 1.3 Format

A markdown table:

```
| # | Prompt | Base | SFT | RLHF |
|---|--------|------|-----|------|
| 1 | ...    | ...  | ... | ...  |
```

Paste it under a "Results" section in this file.

### 1.4 Worked example: a 5-row toy table

The following table illustrates the kinds of differences to look for. The
outputs are hand-written caricatures of the failure modes each model tends to
show, not real generations.

```
| # | Prompt                              | Base                                                      | SFT                                            | RLHF                                       |
|---|-------------------------------------|-----------------------------------------------------------|------------------------------------------------|--------------------------------------------|
| 1 | What is 2+2?                        | What is 2+2? What is 3+3? What is 4+4? ...                | 4.                                             | 4.                                         |
| 2 | Capital of Kenya?                   | The capital of Kenya is a country in East Africa...       | The capital of Kenya is Nairobi.               | Nairobi.                                   |
| 3 | List 3 prime numbers.               | Prime numbers are numbers that are only divisible by ...  | 1. 2  2. 3  3. 5                               | 2, 3, 5.                                   |
| 4 | I feel anxious about a deadline.    | Human: I also feel anxious. Assistant: Me too. Human: ... | I'm sorry to hear that. Try breaking it down.  | That sounds stressful. Two things help...  |
| 5 | Write a JSON object with one key.   | { "key": "value", "key2": "value2", "key3": ...           | { "key": "value" }                             | { "key": "value" }                         |
```

Notes on each row:

- Row 1: base loops; SFT and RLHF answer once. SFT and RLHF agree.
- Row 2: base rambles around the answer; SFT answers in a full sentence; RLHF
  is more concise. Concise output is not always preferable; it depends on the
  prompt.
- Row 3: SFT picked up a numbered-list pattern from training. RLHF often
  prefers shorter formats because they tend to score slightly higher on
  average.
- Row 4: base hallucinates a multi-turn dialogue, the classic SFT-fixes-it
  failure mode. SFT and RLHF stay in role.
- Row 5: format-sensitive. Base often fails to terminate JSON. SFT and RLHF
  emit valid JSON. RLHF that adds polite preamble before the JSON is a common
  reward-hacking pattern that breaks downstream parsing.

A real table that looks nothing like this is also informative. RLHF that
rambles like base, or SFT and RLHF that are indistinguishable on every prompt,
both indicate problems. RLHF that mirrors SFT means the policy did not move;
RLHF that mirrors base means the policy moved away from SFT in the wrong
direction.

Trim very long responses, but do not trim away the failure. A rambling model
should be shown rambling enough that the failure mode is visible. A model
that never emits `<|im_end|>` should be noted explicitly.

---

## 2. Win-rate (Problem 6.2)

### 2.1 Setup

On the same 20 prompts, compare the SFT and RLHF responses pairwise. For each
prompt:

- Present both responses without knowing which is which. Use a randomizer
  that swaps the order with 50% probability and keeps the answer hidden.
- Rate one as "better" (more helpful, more accurate, more aligned with the
  instruction) or declare a tie.

### 2.2 Guarding against position bias

Humans (and LLM-as-judge evaluators) have a systematic preference for
whichever option is shown first. Two mitigations:

- Randomize the order (covered above).
- Run the 20 comparisons once, then re-run them with order reversed, and
  average the results. Agreement across the two passes implies the
  randomizer was sufficient; disagreement implies averaging is needed.

### 2.3 Scoring

    win_rate_RLHF = (RLHF_wins + 0.5 * ties) / 20

Target: at least 0.55. RLHF should be meaningfully better than SFT on a
majority of prompts. A win-rate at or below 0.5 indicates the run failed.
Likely causes:

- Weak reward model.
- `beta` too large; the policy barely moved from SFT.
- Too few PPO iterations.
- Reward hacking that is not visible on these particular prompts.

20 prompts is a smoke test. A 55% win-rate on 20 items is a weak signal,
sufficient only to catch obvious regressions and force manual inspection.
Stronger evidence requires more prompts and the same blinded protocol.

### 2.4 If RLHF didn't win

Record the failure honestly:

- What the RLHF responses look like that SFT's don't.
- What the RLHF responses are missing.
- Which hyperparameter to change first, and why.

Then change it and rerun, if applicable.

The label "RLHF lost" carries no information by itself. "RLHF gives longer
answers that score well but ignore requested JSON formatting" points
directly at length bias in the RM and at format-sensitive prompts.

### 2.5 Worked example: a 20-row tally

A filled-in tally for 20 prompts in which RLHF wins with margin to spare.
Each row records the blinded judgement after both passes (forward and
reversed order) have been collected and reconciled.

```
prompt:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
result:  R  R  T  R  S  R  R  T  R  R  S  R  R  R  T  R  S  R  R  R
```

Counts: `R` (RLHF win) = 13, `T` (tie) = 3, `S` (SFT win) = 4. Plug into the
formula:

    win_rate_RLHF = (13 + 0.5 * 3) / 20 = 14.5 / 20 = 0.725

This clears the 0.55 target. The diagnostic pass then reads every row where
SFT won and looks for a pattern. In this hypothetical, three of the four SFT
wins were prompts that asked for a list, and on those rows the RLHF response
was shorter and dropped one required item. That suggests the RLHF policy is
over-compressing list outputs, possibly because the RM has a slight bias
against long responses on chatty prompts. Note this in `notes/06-eval.md` and
either accept it, raise `beta`, or retrain the RM with explicit
length-balanced pairs.

A second hypothetical, less rosy:

```
prompt:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
result:  R  S  T  S  S  R  T  S  R  S  R  S  T  S  R  S  S  T  S  R
```

Counts: `R` = 5, `T` = 4, `S` = 11. Win rate = `(5 + 2) / 20 = 0.35`. RLHF
lost. Re-read section 2.4 of this note, then start with the most likely
cause for the run (usually too-large `beta`, weak RM, or too few PPO outer
iterations).

---

## 3. The retrospective (Problem 6.3)

One page, answering these questions directly.

The retrospective is technical: name the bug, the symptom, the test that
caught it, and the fix. The catalog of failure modes is the artifact of
value, since it makes the next RLHF implementation easier.

### 3.1 What was the hardest part?

Some candidates from past RLHF projects: aligning `logprobs_old` and
`logprobs_new` in the PPO loop; deciding between shared backbone and separate
value network; getting RM pairwise accuracy above the noise floor; managing
memory with four models loaded at once.

### 3.2 Where did you actually spend the most time?

Count commits and notes. The gap between expected difficulty and actual time
spent identifies where the mental model was weakest. A concept that felt easy
but consumed two days is a candidate for another derivation or a stronger
unit test.

### 3.3 Which gradient check saved you?

For every test in `tests/`, ask whether it caught a real bug, and which.
Note redundant tests as well. The result informs which checks are worth
keeping in future projects.

### 3.4 What would you do differently next time?

Candidates: separate value network; adaptive `beta`; more rollout prompts;
`k_3` instead of `k_1` for the KL penalty; explicit length-bias handling in
the RM; initialize the value head from SFT instead of zero.

### 3.5 What did you get wrong and fix?

Trace through the git history for bugs caught during development:

- Did the SFT mask ever include user turns, omit `<|im_end|>`, or get
  inverted?
- Was there an off-by-one or alignment bug between logprobs and tokens in
  the PPO rollout?
- Was the value loss ever unclipped, or were advantages computed without
  masking?

These are the educational points of the exercise: the actual bugs that
appear when RLHF is built from scratch, and the methods for catching them.

### 3.6 What did you not understand before, and what do you understand now?

Pick one thing. Two paragraphs in the form "I used to think X; now I think
Y." Concrete examples are better than vague ones: "I used to think PPO
clipping just limited ratios; now I understand it clips only
advantage-improving moves." Or: "I used to think masking was a data-loader
detail; now I understand it defines the actual supervised objective."

### 3.7 Worked example: a filled retrospective

The following entry illustrates the level of specificity expected. The
numbers and bugs are illustrative, not from a real run.

> **Hardest part.** Aligning `logprobs_old` and `logprobs_new` in the PPO
> inner loop. The shift between input positions and predicted-next-token
> positions is one offset; the shift between prompt+response tokens and
> response-only positions is another. Drawing both alignments on paper was
> required before the per-token KL plot stopped looking like garbage.
>
> **Where I actually spent the most time.** The reward model. Pairwise
> accuracy was stuck at 53% for three days. The cause was a tokenizer
> mismatch: the chat template in `data_hh.py` was using literal `<|im_end|>`
> text but the RM dataloader had been ported from an earlier version that
> stripped trailing whitespace, so the last-token index pointed one position
> too far left. After fixing it, accuracy jumped to 67% in the next
> training run.
>
> **Gradient check that saved me.** The "flip a masked token, loss
> unchanged" test in `tests/test_grad_sft.py`. It caught a bug where the
> SFT mask was being applied to `input_ids` instead of `labels`, which
> meant the model was being graded on the token *before* the assistant
> content. Loss numbers looked sensible; generations were nonsensical
> until the bug was fixed.
>
> **What I would do differently.** Initialize the value head from a small
> random normal instead of zeros. Zero-init meant the first 50 PPO
> iterations had near-zero advantages, which slowed early learning more
> than expected.
>
> **What I got wrong and fixed.** (1) SFT mask off by one (see above). (2)
> GAE applied without `nonterm` mask on padded rows, which leaked future
> reward across rows that finished early. (3) Used `min` where the value
> loss needed `max`, caught by the gradient-check on the clipped branch.
>
> **What I understand now that I did not before.** "I used to think PPO's
> clip was a regularizer; now I think of it as a piecewise gradient that
> turns off whenever firing it would soften a self-correcting move. The
> clip is asymmetric in a way that only makes sense once the per-token
> gradient table is written by hand."

---

## 4. What to commit to `notes/06-eval.md`

- The 20-row generation table from 6.1.
- The 20-row win-rate tally from 6.2 with scoring notes.
- The retrospective from 6.3.

When this file is complete, the course is complete. The notes should then
be useful without the code open: a future reader should be able to
understand what was trained, how it was evaluated, which results were
convincing, and which parts still need work.
