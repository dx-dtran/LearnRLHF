# 06 — Evaluation

## Purpose

Methodology for Module 6 (Problems 6.1 through 6.3). There is almost no math left
at this stage — the theory is all behind you. This note is about:

1. How to fairly compare three models (base vs SFT vs RLHF).
2. How to compute a win-rate from blinded pairwise annotations.
3. What to write in your retrospective.

Fill this file in with your actual outputs as you run the problems.

Evaluation is where you stop optimizing proxies and read what the model actually says.
Training loss, reward model score, KL, and win-rate are useful, but none of them substitutes
for the question a human cares about: does this assistant follow instructions better than
the previous version?

Base GPT-2 should reveal why instruction tuning is needed. SFT should show what imitation
buys. RLHF should show whether optimizing the learned preference signal improved answers
beyond imitation or mostly changed their style. The side-by-side table is where those
differences become visible.

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

Running both is useful: greedy for a clean model-to-model comparison, sampled generation
for a more realistic view of behavior. Greedy outputs can hide diversity problems. Sampled
outputs can hide regressions behind randomness. Together they give a better picture.

### 1.3 Format

A markdown table like:

```
| # | Prompt | Base | SFT | RLHF |
|---|--------|------|-----|------|
| 1 | ...    | ...  | ... | ...  |
```

Paste it under a "Results" section in this file.

### 1.4 Worked example: a 5-row toy table

What the table should look like, with hand-written outputs that illustrate the kinds
of differences you should be looking for. These are not real generations — they are
caricatures of the failure modes each model tends to show.

```
| # | Prompt                              | Base                                                      | SFT                                            | RLHF                                       |
|---|-------------------------------------|-----------------------------------------------------------|------------------------------------------------|--------------------------------------------|
| 1 | What is 2+2?                        | What is 2+2? What is 3+3? What is 4+4? ...                | 4.                                             | 4.                                         |
| 2 | Capital of Kenya?                   | The capital of Kenya is a country in East Africa...       | The capital of Kenya is Nairobi.               | Nairobi.                                   |
| 3 | List 3 prime numbers.               | Prime numbers are numbers that are only divisible by ...  | 1. 2  2. 3  3. 5                               | 2, 3, 5.                                   |
| 4 | I feel anxious about a deadline.    | Human: I also feel anxious. Assistant: Me too. Human: ... | I'm sorry to hear that. Try breaking it down.  | That sounds stressful. Two things help...  |
| 5 | Write a JSON object with one key.   | { "key": "value", "key2": "value2", "key3": ...           | { "key": "value" }                             | { "key": "value" }                         |
```

What to read out of this:

- **Row 1**: base loops; SFT and RLHF answer once. SFT and RLHF agree.
- **Row 2**: base rambles around the answer; SFT answers in a full sentence; RLHF
  is more concise. Concise is not always better — it depends on the prompt.
- **Row 3**: SFT picked up a numbered-list pattern from training. RLHF often
  prefers shorter formats because they are slightly higher reward on average.
- **Row 4**: base hallucinates a multi-turn dialogue (the classic SFT-fixes-it
  failure mode). SFT and RLHF stay in role.
- **Row 5**: format-sensitive. Base often fails to terminate JSON. SFT and RLHF
  emit valid JSON. Watch for RLHF that adds polite preamble before the JSON; that
  is a common reward-hacking pattern that breaks downstream parsing.

If your real table looks nothing like this — for example, RLHF rambles like base, or
SFT and RLHF are indistinguishable on every prompt — that is information. RLHF that
mirrors SFT means the policy did not move; RLHF that mirrors base means the policy
moved away from SFT in the wrong direction.

Trim extremely long responses, but do not trim away the failure. If a model rambles, show
enough rambling that the failure mode is visible. If a model never emits `<|im_end|>`, note
that explicitly.

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

Treat 20 prompts as a smoke test. A 55% win-rate on 20 items is a weak signal, enough to
catch obvious regressions and force manual inspection. For stronger evidence, increase the
prompt count and keep the annotation protocol blinded.

### 2.4 If RLHF didn't win

Don't paper over it. Write down honestly:

- What the RLHF responses look like that SFT's don't.
- What the RLHF responses are missing that you'd want.
- Which hyperparameter you'd change first, and why.

Then actually change it and try again, if you want to.

"RLHF lost" tells you nothing. "RLHF gives longer answers that score well but ignore
requested JSON formatting" points you straight at checking reward length bias and
format-sensitive prompts. Make the failed evaluation specific.

### 2.5 Worked example: a 20-row tally

A filled-in tally for 20 prompts where RLHF wins with margin to spare. Each row
records your blinded judgement after both passes (forward and reversed order) have
been collected and reconciled.

```
prompt:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
result:  R  R  T  R  S  R  R  T  R  R  S  R  R  R  T  R  S  R  R  R
```

Counts: `R` (RLHF win) = 13, `T` (tie) = 3, `S` (SFT win) = 4. Plug into the formula:

    win_rate_RLHF = (13 + 0.5 * 3) / 20 = 14.5 / 20 = 0.725

That clears the 0.55 target. Now do the diagnostic pass: read every row where SFT won
and look for a pattern. In this hypothetical, three of the four SFT wins were prompts
that asked for a list, and on those rows the RLHF response was shorter and dropped one
required item. That is information: your RLHF policy may be over-compressing list
outputs, possibly because the RM has a slight bias against long responses on chatty
prompts. Note this in `notes/06-eval.md` and either accept it, raise `beta`, or
retrain the RM with explicit length-balanced pairs.

A second hypothetical, less rosy:

```
prompt:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
result:  R  S  T  S  S  R  T  S  R  S  R  S  T  S  R  S  S  T  S  R
```

Counts: `R` = 5, `T` = 4, `S` = 11. Win rate = `(5 + 2) / 20 = 0.35`. RLHF lost. Read
section 2.4 of this note again, then start with the most likely cause for your run
(usually too-large `beta`, weak RM, or too few PPO outer iterations).

---

## 3. The retrospective (Problem 6.3)

One page, answering these six questions directly.

Keep the retrospective technical. Name the bug, the symptom, the test that caught it, and the
fix. The point is to make the next RLHF implementation easier because you have a written
catalog of failure modes.

### 3.1 What was the hardest part?

Some candidates from past RLHF projects: aligning `logprobs_old` and
`logprobs_new` in the PPO loop; deciding between shared backbone and separate
value network; getting RM pairwise accuracy above the noise floor; managing
memory with four models loaded at once.

### 3.2 Where did you actually spend the most time?

Count your commits and your notes. The gap between "how hard I thought it would
be" and "how long it actually took" is diagnostic. Almost always, the answer
reveals where your mental model was the weakest.

Time spent is a better teacher than confidence. If a concept felt easy but consumed two days,
that concept deserves another derivation or a better unit test.

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

Pick one thing. Write two paragraphs in the form "I used to think X; now I think Y." If you
can't find one, redo the course. But you *will* find one.

Good examples are concrete: "I used to think PPO clipping just limited ratios; now I
understand it clips only advantage-improving moves." Or: "I used to think masking was a
data-loader detail; now I understand it defines the actual supervised objective."

### 3.7 Worked example: a filled retrospective

Use this as a calibration target for the level of specificity expected. The numbers
and bugs below are illustrative, not from a real run.

> **Hardest part.** Aligning `logprobs_old` and `logprobs_new` in the PPO inner loop.
> The shift between input positions and predicted-next-token positions is one offset;
> the shift between prompt+response tokens and response-only positions is another.
> I had to draw both alignments on paper before the per-token KL plot stopped looking
> like garbage.
>
> **Where I actually spent the most time.** The reward model. Pairwise accuracy was
> stuck at 53% for three days. The cause was a tokenizer mismatch: the chat template
> in `data_hh.py` was using literal `<|im_end|>` text but the RM dataloader had been
> ported from an earlier version that stripped trailing whitespace, so the last-token
> index pointed one position too far left. After fixing it, accuracy jumped to 67%
> in the next training run.
>
> **Gradient check that saved me.** The "flip a masked token, loss unchanged" test in
> `tests/test_grad_sft.py`. It caught a bug where the SFT mask was being applied to
> `input_ids` instead of `labels`, which meant the model was being graded on the
> token *before* the assistant content. Loss numbers looked sensible; generations
> were nonsensical until I fixed it.
>
> **What I would do differently.** Initialize the value head from a small random
> normal instead of zeros. Zero-init meant the first 50 PPO iterations had near-zero
> advantages, which slowed early learning more than I expected.
>
> **What I got wrong and fixed.** (1) SFT mask off by one — see above. (2) GAE
> applied without `nonterm` mask on padded rows, which leaked future reward across
> rows that finished early. (3) Used `min` where the value loss needed `max`,
> caught by the gradient-check on the clipped branch.
>
> **What I understand now that I did not before.** "I used to think PPO's clip was
> a regularizer; now I think of it as a piecewise gradient that turns off whenever
> firing it would soften a self-correcting move. The clip is asymmetric in a way
> that only makes sense once you write the per-token gradient table by hand."

---

## 4. What to commit to `notes/06-eval.md`

- The 20-row generation table from 6.1.
- The 20-row win-rate tally from 6.2 with your scoring notes.
- The retrospective from 6.3.

This is the last file in the curriculum. When it's complete, the course is complete.

The notes should then be useful without the code open. A future reader should be able to
understand what was trained, how it was evaluated, which results were convincing, and which
parts still need work.
