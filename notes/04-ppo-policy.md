# 04 — PPO: clipped surrogate, value loss, entropy

## Purpose

Theory packet for Problems 4.5, 4.6, and 4.7. Derive and reason about the three
terms that make up the PPO loss:

$$
L_{\mathrm{PPO}} = L_{\mathrm{policy}} + c_v \cdot L_{\mathrm{value}} - c_{\mathrm{ent}} \cdot H
$$

Or in code-style:

    L_PPO = L_policy + c_v * L_value - c_entropy * H

By the end you should be able to:

1. State the PPO clipped policy loss and derive its piecewise gradient.
2. Explain why clipping the importance ratio gives us a "trust region" without
   actually solving a constrained optimization problem.
3. Derive the clipped value loss and explain why the clip helps early in training.
4. Derive the gradient of the masked entropy term.

---

## 1. Setup: importance-sampled policy gradient

Recall from `04-ppo-gae.md` that the policy gradient with advantage baseline is:

$$
\nabla_\theta J = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A_t \right]
$$

Or in code-style:

    grad J = E_{tau ~ pi_theta} [ sum_t grad log pi_theta(a_t | s_t) * A_t ]

**The problem.** To estimate this gradient, we need trajectories from the *current*
policy `pi_theta`. But rolling out a batch of trajectories is expensive — each
rollout requires a full autoregressive generation. We'd really like to reuse each
batch of rollouts for several gradient steps to amortize that cost. Once the
policy has been updated a few times, though, the rollouts are "off-policy" and the
expectation is no longer correct for the new policy.

**The fix.** Importance sampling. The rollouts come from a snapshot policy
$\pi_{\mathrm{old}}$, frozen at the start of the current outer iteration. We rewrite
the objective as an expectation under $\pi_{\mathrm{old}}$ with an importance weight:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_{\mathrm{old}}}\left[ \sum_t \rho_t(\theta) \cdot A_t \right]
$$

where the **importance ratio** is:

$$
\rho_t(\theta)
 = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\mathrm{old}}(a_t \mid s_t)}
 = \exp\bigl(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\mathrm{old}}(a_t \mid s_t)\bigr)
$$

Or in code-style:

    J(theta) = E_{tau ~ pi_old} [ sum_t rho_t(theta) * A_t ]
    rho_t(theta) = pi_theta(a_t|s_t) / pi_old(a_t|s_t)
                 = exp(log pi_theta(a_t|s_t) - log pi_old(a_t|s_t))

As long as $\pi_\theta$ stays close to $\pi_{\mathrm{old}}$, the gradient of this
rewritten expression equals the true policy gradient.

**The catch.** If `pi_theta` drifts far from `pi_old`, the ratios `rho_t` can
become enormous or tiny, and the gradient estimator explodes with variance. Every
importance-sampling method has this failure mode. We need some way to keep using
the batch while preventing the policy from moving too far per update.

---

## 2. The clipped surrogate (Schulman et al. 2017)

### 2.1 Definition

For each token, define two candidate surrogates:

$$
\mathrm{surr}_1 = \rho \cdot A,
\qquad
\mathrm{surr}_2 = \mathrm{clip}(\rho, 1 - \varepsilon, 1 + \varepsilon) \cdot A
$$

where $\mathrm{clip}(x, \text{lo}, \text{hi})$ pins $x$ to the range
$[\text{lo}, \text{hi}]$. $\varepsilon$ is typically 0.2.

Or in code-style:

    surr1 = rho * A
    surr2 = clip(rho, 1 - epsilon, 1 + epsilon) * A

The PPO per-token loss is:

$$
L_{\mathrm{clip},t} = -\min(\mathrm{surr}_1, \mathrm{surr}_2)
$$

And the total policy loss, normalized by the number of valid tokens:

$$
L_{\mathrm{policy}} = \frac{1}{N} \sum_{b,t} m_{b,t} \cdot L_{\mathrm{clip},t}
$$

Or in code-style:

    L_clip_t = -min(surr1, surr2)
    L_policy = (1 / N) * sum over (b, t) of mask[b, t] * L_clip[b, t]

where `N` is the count of masked-in response tokens.

Read the `min` carefully: we take the **smaller** (more pessimistic) of the two
surrogates, then *negate* the whole thing for the loss. Equivalently: inside the
objective (before the negation), we take the *minimum* of `surr1` and `surr2`,
i.e. the less favorable one. This is the key idea — the surrogate is set up so
that the policy cannot be rewarded for running the ratio far outside
`[1 - eps, 1 + eps]` in the direction of increasing the objective.

### 2.2 What "pessimistic" means, case by case

There are essentially four cases based on the sign of `A` and where `rho` sits.
Here's a compact table. "Gradient" means gradient of `L_clip_t` with respect to
`log pi_theta(a_t | s_t)`.

| `A` sign | `rho` position | which surrogate is smaller? | `min` picks | gradient |
|----------|----------------|------------------------------|-------------|----------|
| A > 0 | rho inside `[1-eps, 1+eps]` | equal | either | `-A * rho` |
| A > 0 | rho > 1 + eps | `surr2` (clipped is smaller, since `rho*A` is bigger) | `surr2` | **0** (clipped) |
| A > 0 | rho < 1 - eps | `surr1` (unclipped is smaller) | `surr1` | `-A * rho` |
| A < 0 | rho inside `[1-eps, 1+eps]` | equal | either | `-A * rho` |
| A < 0 | rho < 1 - eps | `surr2` | `surr2` | **0** (clipped) |
| A < 0 | rho > 1 + eps | `surr1` | `surr1` | `-A * rho` |

The pattern: **the clip only fires when firing it would pull the objective back**.
That is, when the policy is already moving in a good direction and has gone "too
far". When the policy is moving in the wrong direction (ratio on the wrong side
of 1), the clip *doesn't* fire, so we still get gradient signal pushing the ratio
back toward 1.

### 2.3 Deriving the piecewise gradient

For the unclipped case, the per-token loss is $L = -\rho A$. Using
$\rho = \exp(\log \pi_\theta - \log \pi_{\mathrm{old}})$:

$$
\frac{\partial \rho}{\partial \log \pi_\theta} = \rho
$$

So by the chain rule:

$$
\frac{\partial L}{\partial \log \pi_\theta} = -A \cdot \rho
$$

Or in code-style:

    d rho / d log pi_theta = rho
    d L / d log pi_theta = -A * rho

At $\rho = 1$ (on-policy limit), this simplifies to $-A$, which recovers the
vanilla policy gradient direction.

For the clipped case, $\mathrm{surr}_2 = (\text{constant w.r.t. } \theta) \cdot A$.
The clip saturates, so changing $\log \pi_\theta$ doesn't change $\mathrm{surr}_2$.
Therefore:

$$
\frac{\partial L}{\partial \log \pi_\theta} = 0
$$

Or in code-style:

    d L / d log pi_theta = 0

That token contributes zero signal to this gradient step.

Verify this with an **edge test** in Problem 4.5. Set every ratio to a large
value (say `rho = 5`) and `A > 0`. Every token should land in the "A > 0,
rho > 1 + eps, clip active" regime. Autograd must report exactly zero gradient on
those tokens. If any token has a nonzero gradient, you've got a `max` instead of
`min`, or a sign error somewhere.

### 2.4 Why this works as a trust region

In practice, on a given update, some fraction of tokens hit the clip — this is
the **clip fraction**, typically 10–30% in a healthy run. Clipped tokens
contribute zero to the gradient, so the effective magnitude of the update
*decreases* as `pi_theta` drifts further from `pi_old`. That gives us a soft
self-regulating trust region: small updates move freely, large ones get throttled
automatically.

Contrast with TRPO, PPO's predecessor: TRPO solves a hard constrained
optimization at every step with conjugate-gradient. PPO gets comparable
performance with `min(surr1, surr2)` and a standard optimizer. That simplicity is
why everyone uses PPO.

---

## 3. Value loss (Problem 4.6)

### 3.1 Unclipped form

The value head's job is to regress toward the GAE return:

$$
R_t = A_t + V_{\mathrm{old}}(s_t)
$$

which is computed at rollout time and treated as a constant during optimization
(stop-gradient). Or in code-style:

    R_t = A_t + V_old(s_t)

The simplest value loss is plain MSE:

$$
L_{V,\mathrm{unclipped}} = \frac{1}{2N} \sum_{b,t} m_{b,t} \cdot \bigl(V_\theta(s_t) - R_t\bigr)^2
$$

Or in code-style:

    L_V_unclipped = (1 / (2*N)) * sum over (b, t) of mask[b, t] * (V_theta(s_t) - R_t)^2

Gradient is `mask * (V_theta - R) * dV_theta/dtheta` — standard MSE.

### 3.2 Clipped form

The clipped value loss mirrors the policy clip. Define:

$$
V_{\mathrm{clipped}}(s_t) = V_{\mathrm{old}}(s_t) + \mathrm{clip}\bigl(V_\theta(s_t) - V_{\mathrm{old}}(s_t), -\varepsilon_v, +\varepsilon_v\bigr)
$$

Then:

$$
\mathrm{per\text{-}tok loss} = \tfrac{1}{2} \max\left((V_\theta - R)^2, (V_{\mathrm{clipped}} - R)^2\right)
$$

$$
L_V = \frac{1}{N} \sum_{b,t} m_{b,t} \cdot \mathrm{per\text{-}tok loss}_{b,t}
$$

Or in code-style:

    V_clipped(s_t) = V_old(s_t) + clip(V_theta(s_t) - V_old(s_t), -eps_v, +eps_v)
    per_tok_loss = 0.5 * max( (V_theta - R)^2, (V_clipped - R)^2 )
    L_V = (1 / N) * sum over (b, t) of mask[b, t] * per_tok_loss[b, t]

Same idea as the policy clip: the value can only move at most `eps_v` away from
its snapshot value `V_old` per update. Why **max** here (instead of `min` in the
policy case)? Because squared error is a loss we want to *minimize*, and we want
to pessimistically pick the larger squared error — i.e. the worse of the two
predictions — as our training loss. That keeps `V_theta` from jumping too far in
any single update.

### 3.3 Why clip the value at all

Early in PPO training, the policy is barely moving while the value head is
learning fast from scratch. Without clipping, the value head can overshoot —
predicting return values that don't match the scale of actual future returns —
and this wrecks the advantage estimates, since the TD errors `delta_t` become
garbage. Then the policy loss feeds on garbage advantages and everything
destabilizes.

Clipping `V_theta` to stay within `eps_v` of its previous value keeps advantages
on a stable scale across updates, at the cost of slower value learning.

In the on-policy limit (when `V_theta = V_old`), both branches of the max are
equal and the clip has no effect. As `V_theta` drifts, the clip kicks in.

### 3.4 Gradient of the clipped value loss

Piecewise linear in `V_theta`:

- Where `|V_theta - V_old| <= eps_v`: the clip is inactive,
  `V_clipped = V_theta`, and the two branches of the max are equal. The gradient
  is `mask * (V_theta - R)`.
- Where the clip is active:
  - The clipped branch `(V_clipped - R)^2` has no gradient through `V_theta`
    (because `V_clipped` is pinned).
  - The unclipped branch `(V_theta - R)^2` has the usual gradient.
  - Whichever branch gets picked by `max` determines the gradient.

Gradient-check the whole thing at fp64 in Problem 4.6. Include a case where
`V_theta` is far from both `V_old` and `R` — make sure the gradient matches what
you'd compute by hand for the selected branch.

---

## 4. Entropy bonus (Problem 4.7)

### 4.1 Purpose

Add the negated entropy of the policy to the loss:

$$
L_{\mathrm{total}} = L_{\mathrm{policy}} + c_v \cdot L_{\mathrm{value}} - c_{\mathrm{ent}} \cdot H(\pi_\theta)
$$

Or in code-style:

    L_total = L_policy + c_v * L_value - c_entropy * H(pi_theta)

The negative sign means a gradient step *increases* entropy — rewarding the policy
for keeping its per-token distributions broad. This prevents **premature mode
collapse**, where the policy becomes almost deterministic (always picks the
argmax) early on and stops exploring alternatives.

In our config, `c_entropy = 0.0` by default — we start without an entropy bonus.
If you see the policy go deterministic within a few iterations (entropy dropping
near zero, generations getting repetitive), bump it to `0.01`.

### 4.2 Masked entropy

For one position, the entropy of the per-token distribution $p$ over the vocab is:

$$
H(p) = -\sum_v p_v \log p_v
$$

For a batch of positions:

$$
H_{\mathrm{batch}} = \frac{1}{N} \sum_{b,t} m_{b,t} \cdot H(p_{b,t})
$$

Or in code-style:

    H(p) = - sum over v of p[v] * log p[v]
    H_batch = (1 / N) * sum over (b, t) of mask[b, t] * H(p[b, t])

### 4.3 Gradient of $H$ with respect to logits

Let $z$ be the logits and $p = \mathrm{softmax}(z)$. Start from the definition:

$$
H = -\sum_v p_v \log p_v
$$

Differentiating $p_v \log p_v$ with respect to $z_u$ gives
$(\partial p_v / \partial z_u) \cdot (\log p_v + 1)$ (product rule, using
$\partial \log p_v / \partial p_v = 1/p_v$). So:

$$
\frac{\partial H}{\partial z_u}
 = -\sum_v \frac{\partial p_v}{\partial z_u} \cdot \bigl(\log p_v + 1\bigr)
$$

Now plug in the softmax derivative
$\partial p_v / \partial z_u = p_v(\delta_{v,u} - p_u)$ from `02-sft.md`:

$$
\frac{\partial H}{\partial z_u}
 = -\sum_v p_v (\delta_{v,u} - p_u) \cdot (\log p_v + 1)
$$

Split the $(\delta_{v,u} - p_u)$ into the two pieces. The $\delta_{v,u}$ piece picks
out only the $v = u$ term. The $-p_u$ piece factors out of the sum:

$$
\frac{\partial H}{\partial z_u}
 = - p_u (\log p_u + 1) + p_u \sum_v p_v (\log p_v + 1)
$$

The sum on the right equals $-H + 1$ (since $\sum_v p_v \log p_v = -H$ and
$\sum_v p_v = 1$). Substitute:

$$
\frac{\partial H}{\partial z_u}
 = -p_u (\log p_u + 1) + p_u \cdot (-H + 1)
 = -p_u (\log p_u + H)
$$

In vector form:

$$
\nabla_z H = -p \odot (\log p + H)
$$

(elementwise product). Or in code-style:

    d H / d z[u]
    = - sum over v of (d p[v] / d z[u]) * (log p[v] + 1)
    = - sum over v of p[v] * (delta(v, u) - p[u]) * (log p[v] + 1)
    = - p[u] * (log p[u] + 1) + p[u] * sum over v of p[v] * (log p[v] + 1)
    = - p[u] * (log p[u] + H)     # since sum_v p[v]*log p[v] = -H and sum_v p[v] = 1

    d H / d z  =  - p * (log p + H)

### 4.4 Sanity check

- **Uniform distribution** (`p[v] = 1/V` for all `v`): `log p[v] = -log V`,
  `H = log V`, so `log p + H = 0`. The gradient is zero everywhere, meaning
  entropy is stationary. Good — uniform is the maximum-entropy distribution, and
  we'd expect zero gradient at the maximum.
- **Near-deterministic** distribution: for the wrong classes, `log p[v]` is very
  negative and the gradient is highly nonzero, pointing in a direction that
  smooths the distribution back out. Good.

### 4.5 Test

Gradient-check the entropy at fp64 on a small logits tensor. Apply the mask,
verify that flipping logits *outside* the mask doesn't change the computed
entropy or its gradient.

---

## 5. The full PPO loss

Combine the three pieces:

    L_PPO = L_policy + c_v * L_value - c_entropy * H

Typical coefficients (our config):

- `c_v = 0.5`: value loss has half the weight of the policy loss.
- `c_entropy = 0.0` to start; raise to `0.01` if entropy collapses.
- `epsilon = 0.2` for the policy ratio clip.
- `epsilon_v = 0.2` for the value clip.

Backward through the sum, clip the gradient norm at 1.0, step the optimizer. Four
epochs over the rollout batch (Problem 5.3) before you throw away the rollout and
start the next outer iteration.

---

## 6. What to commit to `notes/04-ppo-policy.md`

After Problems 4.5, 4.6, 4.7, add:

- Your own derivation of the clipped-surrogate piecewise gradient (the table in
  §2.2, but with the algebra written out).
- Your own derivation of the entropy gradient (section 4.3 redone by hand).
- The numerical result of your edge test: "with every ratio set to 5 and A > 0,
  each masked token's gradient was exactly 0.0".
- A note on what coefficients you tried and what you saw happen.
