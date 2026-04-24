# 04 — KL penalty and KL estimators

## Purpose

Theory packet for Problems 4.2 and 4.3 (per-token KL and reward shaping). Short note,
but the KL term is the single biggest cause of "why is PPO unstable?" problems on
text. Read carefully.

By the end you should be able to:

1. Say why the KL penalty is there at all.
2. Derive three single-sample estimators of KL divergence (`k_1`, `k_2`, `k_3`).
3. Explain which estimator we use for the reward penalty, which we use for logging,
   and why.
4. Write down the per-token reward shaping used in InstructGPT-style PPO.

---

## 0. A five-minute KL divergence primer

Before we use KL divergence, we should know what it is. If you already do, skim
this and keep going.

KL divergence is a number that measures **how different two probability
distributions are**, from the point of view of one of them. For two distributions
$P$ and $Q$ over the same set of outcomes, the KL divergence from $P$ to $Q$ is:

$$
\mathrm{KL}(P \| Q) = \sum_x P(x) \cdot \log \frac{P(x)}{Q(x)}
$$

Or in code-style:

    KL(P || Q) = sum over x of P(x) * log( P(x) / Q(x) )

Five facts to internalize:

1. **It is always non-negative.** $\mathrm{KL}(P \| Q) \ge 0$, with equality iff
   $P = Q$ everywhere. (This follows from Jensen's inequality applied to the
   concave function $\log$.)
2. **It is not symmetric.** $\mathrm{KL}(P \| Q) \ne \mathrm{KL}(Q \| P)$ in
   general. Despite the name "divergence", it is not a proper distance.
3. **Units are nats**, because we used natural log. Multiply by
   $1/\ln 2 \approx 1.44$ to convert to bits.
4. **It's an expectation under $P$.** $\sum_x P(x) \log(P(x)/Q(x))$ can be read
   as $\mathbb{E}_{x \sim P}[\log(P(x)/Q(x))]$ — the average log-ratio, averaged
   over samples drawn from $P$. This is important because PPO only has samples
   from one distribution.
5. **It equals zero iff the two distributions agree everywhere.** The smallest
   disagreement gives a strictly positive number.

A tiny worked example. Say the vocabulary has three outcomes. Policy $\pi$ thinks
the probabilities are $(0.5,\, 0.3,\, 0.2)$. Reference $\pi_{\mathrm{ref}}$ thinks
$(0.4,\, 0.4,\, 0.2)$. Plug in:

$$
\mathrm{KL}(\pi \| \pi_{\mathrm{ref}})
= 0.5 \log\tfrac{0.5}{0.4} + 0.3 \log\tfrac{0.3}{0.4} + 0.2 \log\tfrac{0.2}{0.2}
\approx 0.5(0.223) + 0.3(-0.288) + 0.2(0)
\approx 0.025 \text{ nats}
$$

Very small. The two distributions are nearly identical. KL grows fast as
distributions pull apart: if $\pi_{\mathrm{ref}}$ put zero probability on some
outcome that $\pi$ puts positive probability on, the log-ratio would blow up and
KL would be infinite. That's why we can't start PPO from random weights; the
KL penalty would be meaningless.

Intuition for why KL shows up in RLHF: the SFT policy $\pi_{\mathrm{ref}}$ is our
"sane, coherent English" distribution. Any movement of $\pi_\theta$ away from it
is reported as a KL number, and we can put a price on that movement. The
optimization then decides, at every token, whether chasing extra reward is worth
the KL cost.

---

## 1. Why a KL penalty?

The reward model is an imperfect proxy for human judgment. If we optimize it
naively with a strong policy, the policy will find *adversarial high-reward
outputs* — text the RM scores high but humans actually dislike. Think sycophantic
phrasing, confident-sounding nonsense, or specific fingerprints the RM happens to
love. This failure mode is called **reward hacking**.

The fix: penalize the policy for wandering too far from where it started (the SFT
model), measured in KL divergence. The RL objective becomes:

$$
J_{\mathrm{RLHF}}(\theta)
 = \mathbb{E}\left[ r_{\mathrm{RM}}(\tau) - \beta \sum_t \mathrm{KL}\bigl(\pi_\theta(\cdot \mid s_t) \| \pi_{\mathrm{ref}}(\cdot \mid s_t)\bigr) \right]
$$

Or in code-style:

    J_RLHF(theta) = E[ r_RM(tau) - beta * sum over t of KL(pi_theta(.|s_t) || pi_ref(.|s_t)) ]

$\beta$ controls the strength of the penalty:

- Small `beta`: policy is free to explore, but reward-hacking risk is high.
- Large `beta`: policy stays close to SFT, but reward gains will be small.

Two different ways to understand this penalty, both true:

- **Trust region.** `beta` implements a soft trust region that keeps `pi_theta`
  in a neighborhood of `pi_ref`, where the RM's judgments are more likely to be
  reliable (because that's where the RM was trained).
- **Regularization toward a prior.** The SFT policy is our prior. We're doing
  reward-maximization-with-a-prior rather than pure reward-maximization. You can
  view the PPO-with-KL objective as approximating the posterior under that prior.

`pi_ref` is a **frozen copy of `sft.pt`**. It never gets updated during PPO.

---

## 2. Why *per-token* KL, not per-episode?

Option A: compute a single KL per episode and tack it onto the final reward.
Option B: distribute the KL penalty across tokens, one per step.

InstructGPT picks option B. The per-token reward becomes:

$$
r_t = -\beta \cdot \mathrm{KL}_t + r_{\mathrm{RM}} \cdot \mathbf{1}[ t = T_{\mathrm{last}} ]
$$

Or in code-style:

    r_t = -beta * KL_t + r_RM * (t == last_response_token)

Two reasons this is better:

- **Credit assignment.** If one specific token drives up the KL, it gets punished
  at that position. GAE then propagates the penalty only as far backward as
  reasonable. With an episode-level penalty, every token shares the blame
  equally, which washes out the signal.
- **Numerical scale.** Per-episode KL is the sum of per-token KLs. Over 256
  tokens that's a much bigger scalar and makes `beta` much harder to tune.

---

## 3. Three KL estimators

Here's the setup. During rollout, we generated a sequence of tokens from
`pi_theta`. For each generated token `a_t`, we have two numbers:

- `log pi_theta(a_t | s_t)`: the log-probability the current policy assigned.
- `log pi_ref(a_t | s_t)`: the log-probability the frozen reference assigned.

We want to estimate the KL divergence between the two policies *at that state*,
defined as:

$$
\mathrm{KL}\bigl(\pi_\theta(\cdot \mid s_t) \| \pi_{\mathrm{ref}}(\cdot \mid s_t)\bigr)
 = \mathbb{E}_{a \sim \pi_\theta}\left[ \log \pi_\theta(a \mid s_t) - \log \pi_{\mathrm{ref}}(a \mid s_t) \right]
$$

Or in code-style:

    KL(pi_theta(.|s_t) || pi_ref(.|s_t))
      = E_{a ~ pi_theta}[ log pi_theta(a|s_t) - log pi_ref(a|s_t) ]

But we don't have the full distribution or an exact expectation. We have exactly
one sample $a_t$ drawn from $\pi_\theta$. What's our best single-sample estimate?

Define the log-ratio for the token we drew:

$$
L_t = \log \pi_\theta(a_t \mid s_t) - \log \pi_{\mathrm{ref}}(a_t \mid s_t),
\qquad
\rho_t = e^{L_t}
$$

Or in code-style:

    L_t   = log pi_theta(a_t | s_t) - log pi_ref(a_t | s_t)
    rho_t = exp(L_t)

Three standard estimators, $k_1$, $k_2$, $k_3$:

### 3.1 $k_1$: the naive log-ratio

$$
k_1  =  L_t
$$

Or in code-style:

    k_1 = L_t

- **Unbiased**: `E[L_t]` under `pi_theta` is literally the KL, by definition. Good.
- Can go **negative** on a single sample. If we happen to draw a token that the
  reference thinks is more likely than the current policy does, `L_t < 0`. That's
  weird-looking for a "divergence" but mathematically fine.
- **High variance.** Single-sample estimator with no variance reduction.

This is what InstructGPT uses as the shaping penalty in the per-token reward. It's
exactly what we use in `shape_reward`.

### 3.2 $k_2$: half-squared log-ratio

$$
k_2  =  \tfrac{1}{2} L_t^2
$$

Or in code-style:

    k_2 = 0.5 * L_t^2

- Always non-negative. Nice.
- **Biased.** It's actually an unbiased estimate of `0.5 * E[L^2]`, which by
  Jensen's inequality is an upper bound on `0.5 * (E[L])^2` but is *not* equal to
  KL in general.
- Rarely used in modern RLHF. Included here for completeness.

### 3.3 $k_3$: Schulman's unbiased-and-nonnegative

$$
k_3 = (\rho_t - 1) - L_t = e^{L_t} - 1 - L_t
$$

Or in code-style:

    k_3 = (rho_t - 1) - L_t
        = exp(L_t) - 1 - L_t

- Always **non-negative**, because $e^x - 1 - x \ge 0$ for all real $x$ (the
  exponential function lies above its tangent at 0, a direct consequence of
  convexity of $e^x$). Equality only when $x = 0$.
- **Unbiased** (with a caveat).
- **Lower variance than $k_1$** in practice: when $k_1$ has a big-magnitude
 negative sample, $k_3$ pulls it back up via the $\rho - 1$ term, and vice versa.

A quick geometric picture: `k_1` is just the raw log-ratio of whatever token you
happened to draw. `k_3` adds a symmetric correction penalizing any deviation of
`rho` from 1 — both upside and downside.

### 3.4 Which do we use for what?

- **Penalty inside the per-token reward (shaping):** `k_1`.
  - It's unbiased, its gradient with respect to `log pi_theta` is just `1`
    (simple), and it matches InstructGPT exactly.
  - Downside: single samples can be negative, so the per-token reward signal is
    noisy. But GAE helps smooth that out.
- **Logging and diagnostics:** `k_3`.
  - Non-negative, so it plots as an interpretable "how far has the policy moved
    from ref" number.
  - Downside: its gradient is `rho`, which is nonlinear — makes it a pain if you
    try to use it *as* the penalty.

In this repo: `kl_k1` goes into `shape_reward`; `kl_k3` is computed for the CSV
log but not used in gradients.

---

## 4. Reward shaping (Problem 4.3)

Putting it all together. After a rollout we have:

- `rm_reward[b]`: the RM's scalar output at the last real response token,
  shape `(B,)`.
- `logprobs_old[b, t]`: per-token log-prob under `pi_theta` at rollout time,
  shape `(B, T_resp)`.
- `ref_logprobs[b, t]`: per-token log-prob under `pi_ref`, same shape.
- `response_mask[b, t]`: 1 on real response tokens, 0 on padding / post-EOS.

Compute the per-token KL:

$$
\mathrm{KL}_{k1}[b, t] = \log \pi_\theta^{\text{old}}(a_t \mid s_t) - \log \pi_{\mathrm{ref}}(a_t \mid s_t)
$$

Or in code-style:

    KL_k1[b, t] = logprobs_old[b, t] - ref_logprobs[b, t]

And the per-token reward:

$$
r[b, t] = -\beta \cdot \mathrm{KL}_{k1}[b, t] \cdot m[b, t] + r_{\mathrm{RM}}[b] \cdot \mathbf{1}[ t = T_{\mathrm{last}}(b) ]
$$

Or in code-style:

    r[b, t] = -beta * KL_k1[b, t] * response_mask[b, t]
              + rm_reward[b] * (t == last_response_token_of_row_b)

The indicator `(t == last_response_token_of_row_b)` is 1 exactly once per row —
at the last real token of that row's response. That's where the terminal RM
reward gets injected.

### Edge cases to verify

- If a row ended early via `<|im_end|>`, that EOS position is the "last real
  token" for that row. After that, `response_mask` is 0, so every term in the
  reward is 0.
- If a row hit `response_max_len` without producing `<|im_end|>`, the last real
  token is just the final position. The RM scores whatever got generated,
  including the truncation.
- When `beta = 0`, the KL shaping vanishes and you should get pure terminal RM
  reward. Make this an explicit unit test.

---

## 5. Adaptive KL (optional)

InstructGPT actually uses an *adaptive* `beta`: it targets a specific KL budget
(e.g. 6 nats total over 256 tokens) and adjusts `beta` multiplicatively after each
iteration based on whether the measured KL overshot or undershot that target.

We default to a **fixed** `beta = 0.02` in this repo because it's simpler and good
enough for a teaching implementation. Adaptive KL is a nice 30-minute extension
once everything else is working — not required.

---

## 6. Common pitfalls

- **KL computed on positions that aren't real.** Padding, or positions after EOS,
  contribute to the mean if you don't mask. Always multiply `KL_t` by the
  response mask before using it.
- **Ref log-probs recomputed with dropout on.** The reference is supposed to be
  deterministic and frozen. If you forget to call `model.eval()` or use
  `torch.no_grad()`, you'll get noisy ref log-probs, which feed directly into the
  KL penalty and make training jittery.
- **`beta` too small.** Reward climbs, KL explodes faster, responses get weird.
  Classic reward hacking — raise `beta`.
- **`beta` too large.** Reward barely moves, entropy collapses, responses look
  like SFT with minor cosmetic differences. Lower `beta`.

---

## 7. What to commit to `notes/04-ppo-kl.md`

After finishing Problems 4.2 and 4.3, add:

- Your own derivation that `exp(x) - 1 - x >= 0` for all `x`. Easy one-liner from
  convexity: `exp(x)` lies above its tangent at `x = 0`.
- A small experiment: pick two fixed discrete distributions, compute their *exact*
  KL analytically, then sample from one and estimate the KL using both `k_1` and
  `k_3`. Verify that both are approximately unbiased and that `k_1` has clearly
  higher variance. This is the cleanest way to internalize the difference.
