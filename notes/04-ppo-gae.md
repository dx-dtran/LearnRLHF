# 04 — PPO: policy gradient, advantages, and GAE

## Purpose

Theory packet for Problem 4.4 (and conceptual backbone for all of Module 4). This file
covers:

1. The policy gradient theorem — the central identity of RL.
2. Why we subtract a baseline — variance reduction.
3. The bias–variance trade-off between one-step TD and full Monte Carlo returns.
4. GAE, the recursion that lets us dial between those two extremes.
5. How all of this plays out for per-token rewards on text.

Related notes: `04-ppo-kl.md` (the KL penalty) and `04-ppo-policy.md` (the PPO
clipped surrogate, value loss, entropy). Read those after this one.

---

## 1. Notation for RL on text

Our "environment" is one episode of assistant generation. Given a prompt `s_0`
(a sequence of tokens), the policy `pi_theta` generates a response token by token:

- **State at time `t`**: `s_t = (prompt, a_0, a_1, ..., a_{t-1})` — everything the
  model has seen and produced so far.
- **Action at time `t`**: `a_t`, the next token, sampled from
  `pi_theta(. | s_t)`.
- **Transition**: deterministic. `s_{t+1} = s_t` with `a_t` appended.
- **Reward**: a per-token signal `r_t` that we *construct* from the RM and a KL
  penalty. We'll define this in `04-ppo-kl.md`. For now just take `r_t` as given.
- **Termination**: the episode ends at time `T` when the policy emits `<|im_end|>`,
  or when it hits the maximum response length.

The objective is the expected total reward over a sampled trajectory:

    J(theta) = E_{tau ~ pi_theta} [ sum_t gamma^t * r_t ]

`gamma` is a discount factor in `[0, 1]`. For text RL we use `gamma = 1` — episodes
are short, and we want credit assignment to flow across the entire response.

---

## 2. The policy gradient theorem

### 2.1 What it says

The gradient of the expected return, with respect to the policy parameters `theta`,
can be written as:

    grad J = E_{tau ~ pi_theta} [ sum_t grad log pi_theta(a_t | s_t) * G_hat_t ]

where `G_hat_t` is any unbiased estimate of "the return from time `t` onward",
meaning `E[ sum over k >= t of gamma^{k-t} * r_k ]`. The simplest choice is the
Monte Carlo return — just the actual cumulative reward we observed from time `t`:

    G_hat_t = sum over k = t..T-1 of gamma^{k-t} * r_k

### 2.2 Where it comes from (sketch)

Do the full derivation on paper at least once. Sketch:

- `J(theta) = sum over trajectories tau of p_theta(tau) * R(tau)`, where `R(tau)`
  is the total return of the trajectory.
- Factor the trajectory probability:
  `p_theta(tau) = P(s_0) * product_t pi_theta(a_t | s_t) * P(s_{t+1} | s_t, a_t)`.
  Only the policy factors depend on `theta`.
- Take `log` and then `grad`: `grad log p_theta(tau) = sum_t grad log pi_theta(a_t | s_t)`.
  The transition probabilities and initial state drop out because they don't depend
  on `theta`.
- Use the log-derivative trick: `grad J = E[ grad log p_theta(tau) * R(tau) ]`.
- Apply causality: action `a_t` can only affect rewards from time `t` onward. So
  inside the sum over `t`, we can replace the total return `R(tau)` with just the
  return-to-go `G_hat_t` without changing the expectation.

### 2.3 What this means intuitively

At each position `t`, for the token `a_t` we actually sampled, we nudge the model to
make that token *more likely* (positive gradient on its log-probability) if the
future went well (`G_hat_t > 0`), and *less likely* if the future went badly. The
size of the nudge is proportional to how much better or worse than zero the future
was.

That's it. That's the policy gradient. It's intuitive and correct.

**But**: Monte Carlo returns `G_hat_t` are *very noisy*. A single trajectory's
cumulative reward depends on many sampled tokens' worth of randomness. So even if
our policy is good, the gradient estimate has huge variance. Lowering that variance
without introducing bias is the whole game from here on.

---

## 3. Baselines and advantages

Here's a nice fact. For any function `b(s_t)` that depends only on the state (not
on the action we took):

    E_{a ~ pi} [ grad log pi(a | s_t) * b(s_t) ]  =  0

Why? Because `b(s_t)` doesn't depend on `a`, so it pulls out of the expectation
over `a`. What's left is `E_a[ grad log pi(a | s_t) ]`, which equals
`grad E_a[ 1 ] = grad 1 = 0`.

So we can *subtract* any state-dependent `b(s_t)` from our return estimate
without changing the expected gradient:

    grad J  =  E[ sum_t grad log pi(a_t | s_t) * (G_hat_t - b(s_t)) ]

This is unbiased for any choice of `b(s_t)`. But a good choice of `b` can slash
the variance of the estimator. The best choice is `b(s_t) = E[G_t | s_t]` — the
expected return starting from state `s_t` — because subtracting it leaves behind
only the *action-specific* deviation from the average.

We learn such a `b` with a neural network and call it the **value function**
`V(s_t)`. The difference between the observed return and the expected return is
called the **advantage**:

    A_t = G_hat_t - V(s_t)

Read "advantage" literally: how much better (or worse) did this action turn out than
what we expected before taking it?

The variance-reduced policy gradient:

    grad J = E[ sum_t grad log pi(a_t | s_t) * A_t ]

Still unbiased. Much lower variance in practice. This is what every modern policy
gradient algorithm uses.

---

## 4. TD error and the bias–variance spectrum

Before we define GAE, one more building block: the **one-step TD error**.

    delta_t  =  r_t + gamma * V(s_{t+1}) - V(s_t)

Read this as: "the reward I actually got, plus what the value function says is left
to earn from the next state, minus what I had expected from this state". It's the
part of the return at time `t` that wasn't predicted by the value function. (TD
stands for "temporal difference".)

We can build advantage estimators out of TD errors, and we can pick how many of
them to use:

### 4.1 One-step advantage

Use one TD error:

    A_t^(0) = delta_t

- **Low variance**: depends on only one reward sample.
- **High bias**: relies heavily on `V(s_{t+1})` being accurate. If `V` is wrong, so
  is this advantage.

### 4.2 Monte Carlo advantage

Sum rewards all the way to the episode end:

    A_t^(infinity) = sum over k = t..T-1 of gamma^{k-t} * r_k  -  V(s_t)

- **Zero bias**: doesn't trust `V` at all except as the baseline at time `t`.
- **High variance**: the tail of rewards is noisy.

### 4.3 n-step in between

You can also use `n` real rewards and then bootstrap:

    A_t^(n) = r_t + gamma * r_{t+1} + ... + gamma^{n-1} * r_{t+n-1} + gamma^n * V(s_{t+n}) - V(s_t)
            = sum over k = 0..n-1 of gamma^k * delta_{t+k}

(The second line is a nice algebraic identity — check it by expanding the definition
of `delta`. The intermediate `V` terms telescope.)

As `n` grows, bias falls and variance rises. GAE gives us a single knob that slides
smoothly between these choices.

---

## 5. Generalized Advantage Estimation (Schulman et al. 2016)

### 5.1 Definition

GAE is an exponentially weighted average of the n-step advantages for all `n >= 1`.
In terms of TD errors:

    A_t^GAE  =  sum over k = 0, 1, 2, ... of (gamma * lambda)^k * delta_{t+k}

The knob is `lambda` in `[0, 1]`:

- `lambda = 0`: GAE collapses to one-step TD, `A_t = delta_t`. Low variance, high
  bias.
- `lambda = 1`: GAE collapses to Monte Carlo. No bias, high variance.
- `lambda` in between: smoothly interpolate. The standard choice for text RL is
  `lambda = 0.95`.

### 5.2 Deriving the recursion

This is the clever trick. Split off the first term of the sum:

    A_t = delta_t + sum over k = 1, 2, ... of (gamma*lambda)^k * delta_{t+k}
        = delta_t + (gamma*lambda) * sum over k = 0, 1, ... of (gamma*lambda)^k * delta_{t+1+k}
        = delta_t + (gamma*lambda) * A_{t+1}

So the GAE advantage satisfies:

    A_t = delta_t + gamma * lambda * A_{t+1}

Boundary condition: at the terminal time `T`, the episode is over, so `V(s_T) = 0`
by convention and `A_T = 0`.

This recursion runs *backwards* through time, from `T-1` down to `0`, and that's
how we compute it in code.

### 5.3 Algorithm

```
A[T]     = 0
V[T]     = 0              # by convention
for t = T-1, T-2, ..., 0:
    delta_t = r[t] + gamma * V[t+1] * nonterm[t+1] - V[t]
    A[t]    = delta_t + gamma * lambda * A[t+1] * nonterm[t+1]
```

The `nonterm[t+1]` factor is how we handle variable-length sequences in a batched
loop. It's 1 if position `t+1` is still part of the real episode, 0 if it's
padding or beyond the end. When `nonterm = 0`, the bootstrap from the future is
zeroed out.

### 5.4 Returns as value targets

After computing advantages, the value head needs a regression target. We use:

    R_t = A_t + V_t

This is sometimes called the "bootstrapped Monte Carlo return". The reasoning: if
`A_t` estimates "how much better the return was than we expected" and `V_t`
estimates "what we expected", then their sum estimates the actual return.

**Important.** When training the value head, `R_t` is treated as a stop-gradient
*constant*, even though we computed it from `V_t` just now. Otherwise the value
head would be chasing its own tail — trying to make `V_t` match something that
itself depends on `V_t`. Nothing would train.

In practice: compute the advantages and returns under `torch.no_grad()` (or detach
the values first). The `ppo_core.gae` function can stay pure; the `train_ppo.py`
driver is responsible for wrapping it correctly.

### 5.5 Unit test (Problem 4.4)

Hand-computed 3-step example. Set `T = 3`, `values = [0, 0, 0]`,
`rewards = [1, 0, 0]`, `mask = [1, 1, 1]`, `gamma = 1`, `lambda = 1`.

Walking the recursion backwards:

- `delta_2 = 0 + 0 - 0 = 0`, so `A_2 = 0`.
- `delta_1 = 0 + 0 - 0 = 0`, so `A_1 = 0 + 1 * 0 = 0`.
- `delta_0 = 1 + 0 - 0 = 1`, so `A_0 = 1 + 1 * 0 = 1`.

Result: `advantages = [1, 0, 0]`, `returns = [1, 0, 0]`.

Add a second test with a pad token in the middle — your implementation should
handle the `nonterm` mask correctly, zeroing out the bootstrap at the pad
position.

---

## 6. Per-token rewards for text

In classical RL, the environment hands you a reward at every step. For text we
have to *construct* the per-token reward ourselves:

    r_t  =  - beta * KL_t(pi_theta || pi_ref)     +     r_RM(y) * (t == last_response_token)

Two pieces:

- A small **per-token KL penalty** that punishes the policy for drifting from the
  frozen reference (SFT) model at every step of the response.
- A **terminal RM reward**, delivered as a single scalar at the last real token of
  the response.

See `04-ppo-kl.md` for the KL penalty details — why it's there, what estimator we
use, and how it's computed.

Two consequences worth understanding:

1. **The per-token reward is not the RM score alone.** The KL penalty is baked in
   at the per-token level, so the policy learns to trade off reward-seeking against
   staying close to the SFT distribution *along the whole response*, not just at
   the end.
2. **GAE propagates the terminal RM reward backward through the response.** The
   advantage at an intermediate token reflects how much the terminal reward
   exceeded expectation, minus the KL cost accumulated since that point. GAE's
   exponential weighting decides how much credit each intermediate token gets for
   the final outcome.

---

## 7. Advantage normalization (Problem 4.8)

Before feeding advantages into the PPO policy loss, normalize them across valid
tokens so they have mean 0 and std 1:

    mean = (sum over (b, t) of mask[b, t] * A[b, t])  /  sum(mask)
    var  = (sum over (b, t) of mask[b, t] * (A[b, t] - mean)^2)  /  sum(mask)
    A_tilde = (A - mean) / (sqrt(var) + epsilon)

Why? Because the PPO clip range (typically `epsilon = 0.2`) was chosen assuming
advantages on roughly unit scale. If the RM happens to output rewards on the scale
of 10 or 100, the advantages will be correspondingly huge, every step will hit the
clip, and no learning happens. Normalizing makes the effective learning rate
invariant to the absolute scale the RM happens to have learned.

**The mask matters.** Compute the mean and variance only over the positions where
the mask is 1 — the real response tokens. Padding tokens are garbage data, and if
you let them into the stats, they'll pollute the normalization. The unit test in
Problem 4.8 inserts huge garbage values at masked positions and asserts that the
normalized output's mean and std are unchanged.

---

## 8. What to commit to `notes/04-ppo-gae.md`

After finishing Problems 4.4 and 4.8, add:

- Your own derivation of the GAE recursion (section 5.2 above, redone by hand).
- The numeric output of your unit tests, including the pad-in-the-middle case.
- A one-line explanation of what `lambda` does. You should be able to say this
  without any hesitation.
