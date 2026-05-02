# 04 — KL penalty and KL estimators

## Purpose

Theory packet for Problems 4.2 and 4.3 (per-token KL and reward shaping).
The KL term is short to write down, but it is the largest single source of
PPO instability on text, and the math behind it gets brushed past in many
references.

In Module 3 a reward model was trained that scores any (prompt, response)
pair. Naively, PPO can update the policy ("policy" = the network being
trained, started from the SFT weights) so it produces responses with the
highest possible reward-model score. The problem is that the reward model
was trained on SFT-style responses, so its scoring is only reliable in a
neighborhood around the SFT distribution. A policy free to drift far from
SFT will discover responses the RM gives high scores to but humans dislike
(repeating sycophantic phrases, weird formatting, exact n-grams the RM
happened to overweight). This phenomenon is called **reward hacking**.

The fix is to add a soft penalty for moving away from the SFT model,
measured as a KL divergence between the policy's per-token distribution
and the SFT model's per-token distribution. The SFT model is held frozen
during PPO and is called the **reference model**, written `pi_ref`. The
penalty is added inside the per-token reward, so every step where the
policy strays from the reference pays a small cost.

This note covers four concrete things:

1. A short primer on KL divergence (§0).
2. Why a KL-to-reference penalty appears at all (§1).
3. Three single-sample estimators of KL (`k_1`, `k_2`, `k_3`); which
   estimator is used for the reward penalty, which is used for logging,
   and why (§3).
4. The full reward-shaping formula PPO uses, combining the RM's terminal
   score with the per-token KL penalty (§4).

---

## 0. A five-minute KL divergence primer

KL divergence is a number that measures how different two probability
distributions are, from the point of view of one of them. For two
distributions $P$ and $Q$ over the same set of outcomes, the KL
divergence from $P$ to $Q$ is:

$$
\mathrm{KL}(P \| Q) = \sum_x P(x) \cdot \log \frac{P(x)}{Q(x)}
$$

Or in code-style:

    KL(P || Q) = sum over x of P(x) * log( P(x) / Q(x) )

Five facts:

1. **It is always non-negative.** $\mathrm{KL}(P \| Q) \ge 0$, with
   equality iff $P = Q$ everywhere. (Follows from Jensen's inequality
   applied to the concave function $\log$.)
2. **It is not symmetric.** $\mathrm{KL}(P \| Q) \ne \mathrm{KL}(Q \|
   P)$ in general. Despite the name "divergence", it is not a proper
   distance.
3. **Units are nats**, since the natural log is used. Multiply by $1/\ln
   2 \approx 1.44$ to convert to bits.
4. **It is an expectation under $P$.** $\sum_x P(x) \log(P(x)/Q(x))$ can
   be read as $\mathbb{E}_{x \sim P}[\log(P(x)/Q(x))]$, the average
   log-ratio over samples drawn from $P$. PPO only has samples from one
   distribution.
5. **It equals zero iff the two distributions agree everywhere.** Any
   disagreement gives a strictly positive number.

The direction matters. $\mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}})$
averages over tokens the current policy actually samples. It answers:
how surprising are my current-policy samples under the reference? The
reverse direction would average over reference samples and answer a
different question. PPO rollouts naturally produce samples from the
current policy, so the forward direction is the convenient one.

A small worked example. Say the vocabulary has three outcomes. Policy
$\pi$ has probabilities $(0.5,\, 0.3,\, 0.2)$. Reference
$\pi_{\mathrm{ref}}$ has probabilities $(0.4,\, 0.4,\, 0.2)$:

$$
\mathrm{KL}(\pi \| \pi_{\mathrm{ref}})
= 0.5 \log\tfrac{0.5}{0.4} + 0.3 \log\tfrac{0.3}{0.4} + 0.2 \log\tfrac{0.2}{0.2}
\approx 0.5(0.223) + 0.3(-0.288) + 0.2(0)
\approx 0.025 \text{ nats}
$$

The two distributions are nearly identical. KL grows quickly as
distributions pull apart: if $\pi_{\mathrm{ref}}$ assigns zero
probability to an outcome that $\pi$ assigns positive probability to,
the log-ratio is infinite, so KL is infinite. PPO cannot start from
random weights, since the KL penalty would then be meaningless.

The role of KL in RLHF is to constrain how far $\pi_\theta$ moves from
$\pi_{\mathrm{ref}}$. The SFT policy $\pi_{\mathrm{ref}}$ is a coherent
distribution over assistant responses, and the KL penalty puts a price
on movement away from it. The optimization decides, at every token,
whether chasing extra reward is worth the KL cost.

The SFT distribution is also the region where the reward model has the
best chance of being meaningful. If PPO wanders into bizarre text that
the RM never saw during training, a high reward score is no longer
evidence of a good answer. KL keeps the search local enough that the
reward signal remains usable.

---

## 1. Why a KL penalty?

The reward model is an imperfect proxy for human judgment. Optimizing it
naively with a strong policy produces adversarial high-reward outputs:
text that the RM scores high but humans actually dislike (sycophantic
phrasing, confident-sounding nonsense, or specific patterns the RM
happens to favor). This failure mode is called **reward hacking**.

The fix: penalize the policy for wandering too far from where it
started (the SFT model), measured in KL divergence. The RL objective
becomes:

$$
J_{\mathrm{RLHF}}(\theta)
 = \mathbb{E}\left[ r_{\mathrm{RM}}(\tau) - \beta \sum_t \mathrm{KL}\bigl(\pi_\theta(\cdot \mid s_t) \| \pi_{\mathrm{ref}}(\cdot \mid s_t)\bigr) \right]
$$

Or in code-style:

    J_RLHF(theta) = E[ r_RM(tau) - beta * sum over t of KL(pi_theta(.|s_t) || pi_ref(.|s_t)) ]

$\beta$ controls the strength of the penalty:

- Small `beta`: policy is free to explore, but reward-hacking risk is
  high.
- Large `beta`: policy stays close to SFT, but reward gains are small.

Two equivalent ways to interpret the penalty:

- **Trust region.** `beta` implements a soft trust region that keeps
  `pi_theta` in a neighborhood of `pi_ref`, where the RM's judgments are
  more likely to be reliable (since that is where the RM was trained).
- **Regularization toward a prior.** The SFT policy is a prior. PPO is
  performing reward maximization with a prior rather than pure reward
  maximization. The PPO-with-KL objective approximates the posterior
  under that prior.

`pi_ref` is a frozen copy of `sft.pt`. It is never updated during PPO.

Freezing the reference is necessary: a moving reference would cause the
KL penalty to chase a moving target and stop measuring distance from
the original supervised model.

---

## 2. Why *per-token* KL, not per-episode?

Option A: compute a single KL per episode and add it onto the final
reward.

Option B: distribute the KL penalty across tokens, one per step.

InstructGPT picks option B. The per-token reward becomes:

$$
r_t = -\beta \cdot \mathrm{KL}_t + r_{\mathrm{RM}} \cdot \mathbf{1}[ t = T_{\mathrm{last}} ]
$$

Or in code-style:

    r_t = -beta * KL_t + r_RM * (t == last_response_token)

Two reasons option B is better:

- **Credit assignment.** If one specific token drives up the KL, it
  gets penalized at that position. GAE then propagates the penalty only
  as far backward as appropriate. With an episode-level penalty, every
  token shares the blame equally and the signal is washed out.
- **Numerical scale.** Per-episode KL is the sum of per-token KLs. Over
  256 tokens that is a much larger scalar, and `beta` becomes much
  harder to tune.

Per-token KL also produces more interpretable logs. Mean KL per token
and total KL per response can be tracked separately. A response with
many low-KL tokens and one huge-KL token is a different failure mode
from a response whose every token is slightly off-distribution.

---

## 3. Three KL estimators

During rollout, a sequence of tokens is sampled from `pi_theta`. For
each generated token `a_t`, two numbers are available:

- `log pi_theta(a_t | s_t)`: the log-probability the current policy
  assigned.
- `log pi_ref(a_t | s_t)`: the log-probability the frozen reference
  assigned.

The goal is to estimate the KL divergence between the two policies at
that state:

$$
\mathrm{KL}\bigl(\pi_\theta(\cdot \mid s_t) \| \pi_{\mathrm{ref}}(\cdot \mid s_t)\bigr)
 = \mathbb{E}_{a \sim \pi_\theta}\left[ \log \pi_\theta(a \mid s_t) - \log \pi_{\mathrm{ref}}(a \mid s_t) \right]
$$

Or in code-style:

    KL(pi_theta(.|s_t) || pi_ref(.|s_t))
      = E_{a ~ pi_theta}[ log pi_theta(a|s_t) - log pi_ref(a|s_t) ]

Only one sample $a_t$ from $\pi_\theta$ is available, not the full
distribution or the exact expectation. The estimators below average to
the true KL across many samples but are not constrained to be
nonnegative on a single sample. Negative single-sample KL values are
visually confusing in training logs, which motivates the separate
diagnostic estimator.

Define the current-policy log-ratio for the token drawn:

$$
L_t = \log \pi_\theta(a_t \mid s_t) - \log \pi_{\mathrm{ref}}(a_t \mid s_t),
\qquad
\rho_t = e^{L_t}
$$

Or in code-style:

    L_t   = log pi_theta(a_t | s_t) - log pi_ref(a_t | s_t)
    rho_t = exp(L_t)

For the nonnegative diagnostic estimator, the inverse ratio is also
needed:

$$
\bar{\rho}_t = e^{-L_t} = \frac{\pi_{\mathrm{ref}}(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)}
$$

Or in code-style:

    inv_rho_t = exp(-L_t) = pi_ref(a_t | s_t) / pi_theta(a_t | s_t)

The samples come from $\pi_\theta$. Under that sampling distribution,
$\mathbb{E}_{a \sim \pi_\theta}[\bar{\rho}_t - 1] = 0$, which is what
keeps the diagnostic unbiased for the forward KL. Substituting $\rho_t -
1$ instead is still nonnegative but no longer estimates the same KL
under current-policy samples.

Three standard estimators, $k_1$, $k_2$, $k_3$:

### 3.1 $k_1$: the naive log-ratio

$$
k_1  =  L_t
$$

Or in code-style:

    k_1 = L_t

- **Unbiased**: `E[L_t]` under `pi_theta` is the KL by definition.
- Can be **negative** on a single sample. If the drawn token is more
  likely under the reference than the current policy, `L_t < 0`. The
  estimator can be negative even though the true KL cannot.
- **High variance**: single-sample estimator with no variance reduction.

This is the estimator InstructGPT uses as the shaping penalty in the
per-token reward. It is what `shape_reward` uses.

`k_1` is just a log-prob difference, so its gradient with respect to the
current log-prob is simple. A token that is more likely under the
current policy than the reference pays a positive cost; a token that is
less likely receives a negative sample contribution. The expectation
over samples is the forward KL.

### 3.2 $k_2$: half-squared log-ratio

$$
k_2  =  \tfrac{1}{2} L_t^2
$$

Or in code-style:

    k_2 = 0.5 * L_t^2

- Always non-negative.
- **Biased.** It is an unbiased estimate of `0.5 * E[L^2]`, which by
  Jensen's inequality is an upper bound on `0.5 * (E[L])^2` but is not
  equal to KL in general.
- Rarely used in modern RLHF; included for completeness.

### 3.3 $k_3$: Schulman's unbiased-and-nonnegative diagnostic

$$
k_3 = (\bar{\rho}_t - 1) + L_t = e^{-L_t} - 1 + L_t
$$

Or in code-style:

    k_3 = (inv_rho_t - 1) + L_t
        = exp(-L_t) - 1 + L_t

- Always **non-negative**, since $e^x - 1 - x \ge 0$ for all real $x$
  (the exponential lies above its tangent at 0, by convexity of $e^x$).
  Set $x = -L_t$. Equality holds only when $L_t = 0$.
- **Unbiased for forward KL under current-policy samples**, because
  $\mathbb{E}_{a \sim \pi_\theta}[\bar{\rho}_t - 1] = 0$, so
  $\mathbb{E}_{a \sim \pi_\theta}[k_3] = \mathbb{E}_{a \sim
  \pi_\theta}[L_t]$.
- **Lower variance than $k_1$** in practice: the inverse-ratio
  correction pulls extreme log-ratio samples back toward a more stable
  nonnegative diagnostic.

`k_1` is the raw log-ratio of the sampled token. `k_3` adds a
correction that is zero in expectation but makes each sample
nonnegative, so plotted values behave like a distance-from-reference
diagnostic instead of a noisy signed signal.

Conventions vary across PPO codebases and Schulman's blog posts. Some
definitions use a ratio `p/q`, others use `q/p`, depending on which
distribution produced the samples and which direction of KL is being
estimated. In this repo, the practical rule is: use `k_1` for the
shaping penalty, and use the nonnegative `k_3` helper as a logging
diagnostic. Do not silently swap one for the other.

### 3.4 Which do we use for what?

- **Penalty inside the per-token reward (shaping):** `k_1`.
  - Unbiased, gradient with respect to `log pi_theta` is just `1`,
    matches InstructGPT exactly.
  - Single samples can be negative, so the per-token reward signal is
    noisy. GAE smooths that out.
- **Logging and diagnostics:** `k_3`.
  - Non-negative, so it plots as an interpretable
    distance-from-reference number.
  - Its gradient is nonlinear in the log-ratio, which makes it
    inconvenient as the penalty itself.

In this repo, `kl_k1` goes into `shape_reward`; `kl_k3` is computed for
the CSV log but does not appear in gradients.

The training reward and the logging metric serve different purposes.
The training reward should match the algorithm being studied. The
logging metric should be easy to read.

### 3.5 Worked example: comparing k_1, k_2, k_3 on the same samples

Pick two fixed three-class distributions:

    pi      = (0.5, 0.3, 0.2)         # current policy
    pi_ref  = (0.4, 0.4, 0.2)         # reference (frozen)

Analytic forward KL (computed as a check; PPO never actually has it):

    KL(pi || pi_ref) = 0.5*log(0.5/0.4) + 0.3*log(0.3/0.4) + 0.2*log(0.2/0.2)
                     ≈ 0.0247 nats

Imagine 5 single-token samples drawn from `pi`. For each sample the
action `a` is recorded, then `L = log pi(a) - log pi_ref(a)` and the
three estimators are computed. Suppose the samples are `a = 0, 0, 1, 0,
2` (plausible under `pi`, since class 0 is most likely).

```
sample  a    pi(a)  pi_ref(a)   L = log(pi/pi_ref)   k_1     k_2 = 0.5*L^2   inv_rho = pi_ref/pi   k_3 = inv_rho - 1 + L
1       0    0.5    0.4         +0.2231              +0.2231  +0.0249         0.800                 -0.200 + 0.2231 ≈ +0.0231
2       0    0.5    0.4         +0.2231              +0.2231  +0.0249         0.800                 +0.0231
3       1    0.3    0.4         -0.2877              -0.2877  +0.0414         1.333                 +0.333 - 0.2877 ≈ +0.0457
4       0    0.5    0.4         +0.2231              +0.2231  +0.0249         0.800                 +0.0231
5       2    0.2    0.2          0.0                  0.0      0.0            1.000                  0.0
```

Sample averages:

    mean k_1 ≈ +0.0769          # signed; matches forward KL only in expectation
    mean k_2 ≈ +0.0232          # always non-negative; biased estimator of KL
    mean k_3 ≈ +0.0190          # always non-negative; unbiased for forward KL

`k_1` overshoots the true 0.0247 here on this small sample because it
weighted the three positive samples more than the one negative. Over
many samples, both `k_1` and `k_3` average to ~0.0247. `k_2` measures
the squared log-ratio rather than KL; it happens to be close in this
example, but does not match KL in general.

Sample 3 illustrates the variance reduction: `k_1 = -0.2877` (negative,
unusual-looking for a divergence), while `k_3 = +0.0457` (always
non-negative). `k_3` adds the `(inv_rho - 1)` correction, which has
expected value 0 under `pi` but pulls extreme single-sample values back
toward 0.

### 3.6 Worked example: per-token KL and reward shaping

Take a 5-token response with `beta = 0.02` and terminal reward `r_RM =
+1.5`. Suppose during rollout the per-token current-policy log-probs and
the reference's log-probs were:

```
pos:               0      1      2      3      4
piece:             "The"  "ans"  "wer"  "is"   "<|im_end|>"
logprobs (theta): -1.20  -1.50  -2.00  -1.10  -0.30
logprobs (ref):   -1.50  -1.40  -2.20  -1.30  -0.50
response_mask:    1      1      1      1      1
last_idx:                                       <-- terminal reward injected here
```

Per-token KL via `k_1`:

    KL_k1[t] = logprobs[t] - ref_logprobs[t]
             = +0.30, -0.10, +0.20, +0.20, +0.20

Per-token reward via the shaping formula:

    r[t] = -beta * KL_k1[t] * response_mask[t] + r_RM * (t == last_idx)

```
pos:        0       1       2       3       4
-beta*KL: -0.0060 +0.0020 -0.0040 -0.0040 -0.0040
+r_RM*1{t==4}: 0     0       0       0     +1.5000
r[t]:    -0.0060 +0.0020 -0.0040 -0.0040 +1.4960
```

Most positions get a tiny negative reward (the policy moved slightly
away from the reference and is paying KL). Position 1 has a small
positive reward because the policy moved *toward* the reference there,
which is a single-sample artifact of the signed `k_1` estimator. The
terminal token carries the RM's verdict on the whole response,
attenuated slightly by its own KL term.

GAE then propagates the final +1.5 backward through the response,
discounted by `(gamma * lambda)`, while keeping the per-position KL
costs local. That is what makes the policy learn to stay close to the
reference except where moving away clearly improved the eventual
reward.

### 3.7 Worked example: edge cases for shaping

Two limits worth checking explicitly.

**`beta = 0`.** Substituting into the shaping formula:

    r[t] = -0 * KL_k1[t] + r_RM * (t == last_idx)
         = r_RM * (t == last_idx)

Every position is 0 except the last, which is the terminal RM reward.
There is no token-level pressure to stay near `pi_ref`. PPO discovers
the highest-RM output regardless of how strange. This is the unit test
for `beta = 0`: `shape_reward` should reduce to a single nonzero entry
per row.

**`pi = pi_ref` exactly.** Then `logprobs == ref_logprobs` at every
position, so `KL_k1[t] = 0` for all `t`. The shaped reward is again
purely terminal:

    r[t] = r_RM * (t == last_idx)

This case occurs by construction in the very first PPO iteration after
loading both policy and reference from `sft.pt`. The KL contribution is
exactly zero on the first batch, and the only signal driving the policy
is the terminal RM reward.

---

## 4. Reward shaping (Problem 4.3)

After a rollout the available tensors are:

- `rm_reward[b]`: the RM's scalar output at the last real response
  token, shape `(B,)`.
- `logprobs_old[b, t]`: per-token log-prob under `pi_theta` at rollout
  time, shape `(B, T_resp)`.
- `ref_logprobs[b, t]`: per-token log-prob under `pi_ref`, same shape.
- `response_mask[b, t]`: 1 on real response tokens, 0 on padding /
  post-EOS.

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

The indicator `(t == last_response_token_of_row_b)` is 1 exactly once
per row, at the last real token of that row's response. That is where
the terminal RM reward is injected.

This construction turns a sequence-level score into a token-level RL
problem. Before the last token, the reward mostly says "stay close to
the reference." At the last token, the reward says "the whole response
was this good according to the RM." GAE then propagates the terminal
information backward through the response according to $\gamma$ and
$\lambda$.

### Edge cases to verify

- A row that ended early via `<|im_end|>` has that EOS position as the
  "last real token" for that row. After that, `response_mask` is 0, so
  every term in the reward is 0.
- A row that hit `response_max_len` without producing `<|im_end|>` has
  the final position as its last real token. The RM scores whatever
  was generated, including the truncation.
- When `beta = 0`, the KL shaping vanishes and the result is pure
  terminal RM reward. This is an explicit unit test.

---

## 5. Adaptive KL (optional)

InstructGPT actually uses an adaptive `beta`: it targets a specific KL
budget (for example, 6 nats total over 256 tokens) and adjusts `beta`
multiplicatively after each iteration based on whether the measured KL
overshot or undershot that target.

This repo defaults to a fixed `beta = 0.02` because it is simpler and
sufficient for a teaching implementation. Adaptive KL is a 30-minute
extension once everything else is working. It is not required.

A fixed beta is easier to reason about: one knob has one meaning.
Adaptive beta adds a controller on top, and debugging then requires
isolating PPO's behavior from the controller's response.

---

## 6. Common pitfalls

- **KL computed on positions that are not real.** Padding, or positions
  after EOS, contribute to the mean unless masked. Always multiply
  `KL_t` by the response mask before using it.
- **Ref log-probs recomputed with dropout on.** The reference is
  supposed to be deterministic and frozen. Forgetting `model.eval()` or
  `torch.no_grad()` produces noisy ref log-probs that flow into the KL
  penalty and make training jittery.
- **`beta` too small.** Reward climbs, KL explodes faster, responses
  get weird. Reward hacking; raise `beta`.
- **`beta` too large.** Reward barely moves, entropy collapses,
  responses look like SFT with minor cosmetic differences. Lower
  `beta`.

When diagnosing KL issues, also inspect the generated text. A scalar KL
number can show that the policy moved; it cannot show whether the
movement improved helpfulness or found a reward-model loophole. Pair
the plot with samples.

---

## 7. What to commit to `notes/04-ppo-kl.md`

After finishing Problems 4.2 and 4.3, add:

- Your own derivation that `exp(-x) - 1 + x >= 0` for all `x`. Set `u
  = -x`, then `exp(u) - 1 - u >= 0` because `exp(u)` lies above its
  tangent at `u = 0`.
- A small experiment: pick two fixed discrete distributions, compute
  their exact KL analytically, then sample from one and estimate the
  KL using both `k_1` and `k_3`. Verify that both are approximately
  unbiased and that `k_1` has clearly higher variance.
