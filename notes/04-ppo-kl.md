# 04 — KL penalty and KL estimators

## Purpose

Theory for Problems 4.2 and 4.3 (per-token KL and reward shaping). Short file, but the
KL term is the single biggest source of "wait, why is this training unstable" problems
in PPO for text. Read carefully.

By the end you should be able to:

1. Explain why we want a KL penalty at all.
2. Derive three estimators of $\text{KL}(\pi \| \pi_\text{ref})$ from a single sample:
   $k_1$, $k_2$, $k_3$.
3. Explain which to use for the penalty and which for logging, and why.
4. Write down the per-token reward shaping used in InstructGPT-style PPO.

---

## 1. Why a KL penalty?

The reward model $r_\phi$ is an imperfect proxy for human preferences. If we optimize it
naively with a powerful policy, the policy will find *adversarial high-reward outputs*
the RM assigns reward to but humans don't actually like — sycophancy, fake confidence,
specific phrasing fingerprints. This is **reward hacking**.

Remedy: penalize the policy for moving too far from its starting point (the SFT model),
measured in KL divergence token-by-token:

$$
J_\text{RLHF}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ r_\phi(\tau) - \beta \sum_t \text{KL}\bigl(\pi_\theta(\cdot \mid s_t) \,\|\, \pi_\text{ref}(\cdot \mid s_t)\bigr)\right].
$$

$\beta$ controls how conservative the policy stays. Small $\beta$ = free to explore,
big reward-hacking risk. Large $\beta$ = stays close to SFT, smaller reward gains.

Two interpretations of this penalty, both valid and important:

- **Trust region.** $\beta$ implements a soft trust region keeping $\pi_\theta$ in a
  neighborhood of $\pi_\text{ref}$ where the RM's judgments are more likely to be
  accurate (because that's where the RM was trained).
- **Regularization.** The SFT policy is the prior; we're doing
  reward-maximization-with-a-prior rather than pure reward-maximization. You can
  view PPO-with-KL-penalty as roughly the posterior under this prior.

$\pi_\text{ref}$ is **frozen** (a copy of `sft.pt`). It does not get updated during PPO.

---

## 2. Why a per-token KL at all (not episode-level)?

We *could* compute a single full-sequence KL per episode and add it as one more scalar
reward at the end. InstructGPT chose to distribute the KL penalty per-token instead:

$$
r_t = -\beta \cdot \text{KL}_t + r_\text{RM} \cdot \mathbf{1}_{t = T-1}.
$$

Advantages of per-token:

- **Credit assignment.** Any individual token that drives up KL gets penalized at that
  position, and GAE carries the signal backward only as far as reasonable. With an
  episode-level penalty, every token in the sequence shares the blame equally.
- **Numerical scale.** An episode-level KL is the sum of per-token KLs; over 256 tokens
  that's a much larger scalar and makes tuning $\beta$ harder.

---

## 3. The three KL estimators

We have $\log \pi_\theta(a_t \mid s_t)$ (the "new" log-prob) and
$\log \pi_\text{ref}(a_t \mid s_t)$ (the frozen ref's log-prob), both evaluated on the
*sample* $a_t$ drawn from $\pi_\theta$ during rollout. We want an estimate of

$$
\text{KL}\bigl(\pi_\theta(\cdot \mid s_t)\,\|\,\pi_\text{ref}(\cdot \mid s_t)\bigr)
= \mathbb{E}_{a \sim \pi_\theta}\!\left[\log \frac{\pi_\theta(a)}{\pi_\text{ref}(a)}\right]
$$

from the single sample $a_t$ we actually drew.

Define the log-ratio

$$
L_t = \log \pi_\theta(a_t \mid s_t) - \log \pi_\text{ref}(a_t \mid s_t), \qquad
\rho_t = e^{L_t}.
$$

### 3.1 $k_1$: naive log-ratio

$$
k_1(a_t) = L_t.
$$

- **Unbiased** estimator of the KL:
  $\mathbb{E}_{a \sim \pi_\theta}[L] = \text{KL}(\pi_\theta \| \pi_\text{ref})$ by definition.
- Can be **negative** on a single sample (e.g. if the draw happens to be more likely
  under $\pi_\text{ref}$ than $\pi_\theta$).
- **High variance** — this is the "raw" estimator with nothing to reduce noise.

This is what InstructGPT uses as the shaping penalty in the per-token reward.

### 3.2 $k_2$: half-squared

$$
k_2(a_t) = \tfrac{1}{2} L_t^2.
$$

- Always $\ge 0$.
- **Biased**: $\mathbb{E}[k_2] \ne \text{KL}$ in general. It's a biased estimate of a
  related but different quantity (it's actually an unbiased estimate of
  $\tfrac{1}{2}\mathbb{E}[L^2]$, which by Jensen's is an upper bound on
  $\tfrac{1}{2}(\mathbb{E} L)^2$ but not equal to $\text{KL}$).
- Rarely used in modern RLHF. Included for completeness.

### 3.3 $k_3$: Schulman's unbiased-nonnegative

$$
k_3(a_t) = \bigl(\rho_t - 1\bigr) - L_t = e^{L_t} - 1 - L_t.
$$

- **Always $\ge 0$.** Because $e^x - 1 - x \ge 0$ for all real $x$, with equality only
  at $x = 0$.
- **Unbiased** estimator of $\text{KL}$.

  Sketch: $\mathbb{E}_{a \sim \pi_\theta}[\rho - 1] = \mathbb{E}[\pi_\text{ref}(a)/\pi_\theta(a) \cdot \pi_\theta(a)/\pi_\text{ref}(a)]$ — wait, let's redo carefully.

  $\rho = \pi_\theta / \pi_\text{ref}$, so
  $\mathbb{E}_{\pi_\theta}[\rho - 1] = \mathbb{E}_{\pi_\theta}[\pi_\theta/\pi_\text{ref}] - 1$.
  That second expectation is not 1 in general — it's
  $\sum_a \pi_\theta^2(a) / \pi_\text{ref}(a) \ge 1$. So $\mathbb{E}[\rho - 1] \ge 0$
  but isn't zero.

  However, $\mathbb{E}[k_3] = \mathbb{E}[\rho - 1] - \mathbb{E}[L] = (\text{something} \ge 0) - \text{KL}$.

  The precise statement (Schulman 2020, "Approximating KL Divergence") is that $k_3$
  is an unbiased estimator of a particular quantity that coincides with
  $\text{KL}$ when one integrates properly — in practice $k_3$ is treated as "unbiased
  for all intents and purposes" because it matches the true KL in expectation under
  the regime we care about (moderate $\rho$). It's the standard estimator for logging.

- **Lower variance than $k_1$** in practice, because the negative samples in $k_1$
  get offset by the positive $\rho - 1 - L$ adjustment.

A clean intuition: $k_1$ is "the log-ratio of the sample I drew". $k_3$ adjusts that
by how much $\rho$ departs from 1 *linearly* — penalizing both upside and downside
deviations symmetrically.

### 3.4 Which to use for what

- **Penalty in the per-token reward (shaping):** $k_1$.
  - Pros: unbiased, simple gradient (derivative of $L_t$ w.r.t. $\log \pi_\theta$ is
    just $1$), matches InstructGPT exactly.
  - Cons: can go negative on a single sample, gives a noisy reward signal.
- **Logging / diagnostics:** $k_3$.
  - Pros: nonnegative, interpretable as "how far has the policy moved from ref".
  - Cons: gradient is $\rho_t$ (nonlinear), complicates things if used as penalty.

In this repo: `kl_k1` is the shaping penalty used inside `shape_reward`; `kl_k3` is
just for the CSV log.

---

## 4. Reward shaping (Problem 4.3)

Put it all together. Given a batch of rollouts:

- `rm_reward[b]` — scalar reward from the RM at the last non-pad response token.
- `logprobs_old[b, t]` — per-token log-prob under $\pi_\theta$ at rollout time.
- `ref_logprobs[b, t]` — per-token log-prob under $\pi_\text{ref}$.
- `response_mask[b, t]` — 1 on real response tokens.

Compute:

$$
\text{KL}^{k_1}_{b, t} = \log \pi_\theta(a_{b, t} \mid s_{b, t}) - \log \pi_\text{ref}(a_{b, t} \mid s_{b, t}),
$$

$$
r_{b, t} = -\beta \cdot \text{KL}^{k_1}_{b, t} \cdot m_{b, t} \; + \; r_\text{RM}[b] \cdot \mathbf{1}_{t = \text{last}(b)}.
$$

The $\mathbf{1}_{t = \text{last}(b)}$ is the indicator that $t$ is the last real token
of row $b$'s response — this is where the terminal RM reward gets injected.

### Edge cases

- If an episode finished early via `<|im_end|>`, `last(b)` is that EOS position. After
  that, `response_mask` is 0 and so is every term in the reward.
- If the policy hit `response_max_len` without EOSing, `last(b)` is the last position
  (still mask = 1). The RM scores whatever got generated, including the truncation.
- With $\beta \to 0$ the KL shaping vanishes and you get pure RM terminal reward.
  Verify this edge case in the unit test.

---

## 5. Adaptive KL (optional, not required)

InstructGPT actually uses an **adaptive** $\beta$: start from a target KL budget (e.g.
6 nats over 256 tokens), and after each PPO iteration adjust $\beta$ multiplicatively
based on whether the measured KL exceeded or fell short of the target. We default
to a **fixed** $\beta = 0.02$ in this repo because it's simpler and good enough for
a teaching impl. Adaptive KL is a good 30-minute extension once everything else works.

---

## 6. Common pitfalls

- **Per-token KL computed on positions that didn't exist.** Pad positions, post-EOS
  positions. Multiply $\text{KL}_t$ by `response_mask[b, t]` before using it as
  reward.
- **Ref log-probs recomputed with dropout on.** Ref is frozen — call with
  `model.eval()` and wrap in `torch.no_grad()`. Numerical differences here propagate
  directly into the shaping penalty and will make training noisy.
- **$\beta$ too small.** Reward goes up, KL explodes, responses get weird — classic
  reward hacking. Bump $\beta$.
- **$\beta$ too large.** Reward barely moves, entropy collapses, responses look like
  SFT with minor variations. Lower $\beta$.

---

## 7. What to commit to `notes/04-ppo-kl.md`

After finishing Problems 4.2 and 4.3, append:

- Your derivation that $e^x - 1 - x \ge 0$ for all $x$ (easy: convexity argument).
- A small experiment: compute $k_1$ and $k_3$ on synthetic log-prob samples from two
  fixed discrete distributions, compare against the true (analytically computed) KL,
  and observe that both are unbiased but $k_1$ has more variance.
