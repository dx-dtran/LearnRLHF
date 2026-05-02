# 04 — PPO: clipped surrogate, value loss, entropy

## Purpose

Theory packet for Problems 4.5, 4.6, and 4.7. Companion to
`04-ppo-gae.md` (which produces per-token advantages from a rollout) and
`04-ppo-kl.md` (which produces the per-token reward).

By this point in the curriculum, a rollout produces, for every token in
every sampled response: a token id, the log-probability under the
*rollout-time* policy, an advantage estimate, a regression target for
the value head, and a mask telling the loss which positions are real
response tokens. The job of the present note is to write down the loss
that turns those tensors into a single scalar, takes a backward pass on
it, and gradient-checks every piece.

That total loss has three parts:

$$
L_{\mathrm{PPO}} = L_{\mathrm{policy}} + c_v \cdot L_{\mathrm{value}} - c_{\mathrm{ent}} \cdot H
$$

Or in code-style:

    L_PPO = L_policy + c_v * L_value - c_entropy * H

Each part is derived below in its own section:

- `L_policy` is the **clipped surrogate policy loss**. It pushes
  positive-advantage tokens toward higher probability and
  negative-advantage tokens toward lower probability, with a clip that
  caps how far the new policy can move from the rollout policy in a
  single update (§2).
- `L_value` is the **value-head regression loss**. The value head was
  used in §4-`gae` to compute advantages; here it is trained to predict
  better in the future (§3).
- `H` is the **entropy of the policy's per-token distribution**. Adding
  it with a negative coefficient encourages the policy to keep
  spreading probability across multiple tokens rather than collapsing
  to a single argmax (§4).

By the end of this note the objectives are:

1. State the PPO clipped policy loss and derive its piecewise gradient.
2. Explain why clipping the importance ratio creates a "trust region"
   without solving a constrained optimization problem.
3. Derive the clipped value loss and explain why the clip helps early
   in training.
4. Derive the gradient of the masked entropy term.

The behavior of these terms is piecewise: some tokens have ordinary
policy-gradient signal, some have zero gradient because the ratio is
clipped, some value predictions use the unclipped branch, and some use
the clipped branch. The tests are designed to force each case to
happen.

---

## 1. Setup: importance-sampled policy gradient

Vanilla policy gradient samples a trajectory from the current policy,
then nudges the parameters so the good actions become more likely. Two
problems:

1. Sampling trajectories is expensive. Each one requires a full
   autoregressive generation of up to 256 tokens, and reusing each
   batch of samples for several gradient steps is desirable.
2. Once a step has been taken, the old samples are no longer drawn
   from the new policy, so the naive gradient is biased.

PPO addresses both with a two-step trick: (a) use importance sampling
to reuse the batch for multiple gradient updates, and (b) clip the
importance ratio so that after a few updates the policy cannot wander
far enough from the rollout policy to make the estimator blow up. The
rest of this note formalizes the trick.

The outer-loop / inner-loop structure is the practical motivation.
Rollout is expensive because generation is autoregressive. With a
rollout batch in hand, several optimizer steps should be extracted
from it. PPO is the compromise that allows reuse without pretending
old samples are still perfectly on-policy.

The policy gradient with advantage baseline is (recall from
`04-ppo-gae.md`):

$$
\nabla_\theta J = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A_t \right]
$$

Or in code-style:

    grad J = E_{tau ~ pi_theta} [ sum_t grad log pi_theta(a_t | s_t) * A_t ]

**The problem.** Estimating this gradient requires trajectories from
the *current* policy `pi_theta`. Rolling out a batch of trajectories is
expensive, and reusing each batch for several gradient steps is
desirable. After a few updates, the rollouts are off-policy and the
expectation is no longer correct for the new policy.

**The fix.** Importance sampling. The rollouts come from a snapshot
policy $\pi_{\mathrm{old}}$, frozen at the start of the current outer
iteration. The objective is rewritten as an expectation under
$\pi_{\mathrm{old}}$ with an importance weight:

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

As long as $\pi_\theta$ stays close to $\pi_{\mathrm{old}}$, the
gradient of this rewritten expression equals the true policy gradient.

The ratio is computed only for the action that was actually sampled.
Entire vocabulary distributions are not compared in the policy loss.
For each response token, PPO asks: under the new policy, how much
more or less likely is the exact token sampled during rollout?

**The catch.** If `pi_theta` drifts far from `pi_old`, the ratios
`rho_t` can become enormous or tiny, and the gradient estimator
explodes with variance. Every importance-sampling method has this
failure mode. Some mechanism is needed to keep using the batch while
preventing the policy from moving too far per update.

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

The total policy loss, normalized by the number of valid tokens:

$$
L_{\mathrm{policy}} = \frac{1}{N} \sum_{b,t} m_{b,t} \cdot L_{\mathrm{clip},t}
$$

Or in code-style:

    L_clip_t = -min(surr1, surr2)
    L_policy = (1 / N) * sum over (b, t) of mask[b, t] * L_clip[b, t]

where `N` is the count of masked-in response tokens.

The `min` takes the smaller (more pessimistic) of the two surrogates,
and the negation in front turns it into a loss. Inside the objective
(before negation) the smaller of `surr1` and `surr2` is chosen, the
less favorable one. This setup prevents the policy from being rewarded
for running the ratio far outside `[1 - eps, 1 + eps]` in the
direction of increasing the objective.

The negation is a common source of confusion. PPO papers usually
describe maximizing the surrogate objective, and PyTorch optimizers
minimize a loss. This repo writes the loss as the negative objective.
When checking signs, ask whether gradient descent on the loss
increases good-token log-probs and decreases bad-token log-probs.

A sanity check with numbers. Set $\varepsilon = 0.2$ so the clip range
is $[0.8,\, 1.2]$. Pick two tokens:

- **Token A (good action).** Advantage $A = +1$ (this action turned
  out better than expected). Current ratio $\rho = 1.5$: the policy
  has moved so that this token is 1.5× more likely than at rollout.
  Compute: $\mathrm{surr}_1 = 1.5$, $\mathrm{surr}_2 = 1.2$. $\min =
  1.2$. Loss $= -1.2$. The gradient is **zero** because the flat
  clipped branch won the $\min$. No further pushing this token
  upward; it has already moved too far.
- **Token B (bad action).** Advantage $A = -1$. Ratio $\rho = 1.5$.
  Compute: $\mathrm{surr}_1 = -1.5$, $\mathrm{surr}_2 = -1.2$. $\min =
  -1.5$. Loss $= +1.5$. The gradient is **non-zero** and pushes
  $\rho$ down, which is the correct direction since this action
  turned out badly. The clip does NOT fire, since firing it would
  soften the response to a bad action that should be suppressed.

PPO is pessimistic only when the policy would otherwise get credit
for a move that has already become extreme. If the move is extreme in
a costly direction, the gradient keeps flowing. For positive
advantages, an extreme improvement means the sampled token became
much more likely. For negative advantages, an extreme improvement
means the sampled token became much less likely. The table below
writes out those cases.

Two more numerical cases finish the picture. Same `epsilon = 0.2`,
clip range `[0.8, 1.2]`:

- **Token C (good action moving the wrong way).** Advantage `A = +1`,
  ratio `rho = 0.5`. The policy made a good action *less* likely than
  at rollout. Compute: `surr1 = 0.5`, `surr2 = clip(0.5, 0.8, 1.2) * 1
  = 0.8`. `min = 0.5`. Loss `= -0.5`. The clip does not fire, since
  firing it would replace the smaller unclipped value (`0.5`) with
  the larger clipped value (`0.8`), the wrong direction for a
  "pessimistic" choice. Gradient through `rho` is `-A * rho = -0.5`
  (nonzero), which after one optimizer step makes the action more
  likely again.
- **Token D (bad action moving the wrong way).** Advantage `A = -1`,
  ratio `rho = 2.0`. The policy made a bad action *more* likely than
  at rollout. Compute: `surr1 = -2.0`, `surr2 = clip(2.0, 0.8, 1.2) *
  (-1) = -1.2`. `min = -2.0`. Loss `= +2.0`. The clip does not fire
  here either: replacing `-2.0` with `-1.2` would *raise* the inner
  objective, the opposite of pessimistic. Gradient through `rho` is
  `-A * rho = +2.0` (nonzero, large), pushing this bad action's
  probability back down. The clip never spares the policy from
  correcting a bad move.

Same pattern, four cases. Whenever the ratio is on the
"fighting-the-advantage" side of 1 (good action being made less
likely, or bad action being made more likely), the clip stays out of
the way and the gradient corrects course.

### 2.2 What "pessimistic" means, case by case

Four cases, organized by the sign of `A` and the position of `rho`.
The "Gradient" column is the gradient of `L_clip_t` with respect to
`log pi_theta(a_t | s_t)`.

| `A` sign | `rho` position | which surrogate is smaller? | `min` picks | gradient |
|----------|----------------|------------------------------|-------------|----------|
| A > 0 | rho inside `[1-eps, 1+eps]` | equal | either | `-A * rho` |
| A > 0 | rho > 1 + eps | `surr2` (clipped is smaller, since `rho*A` is bigger) | `surr2` | **0** (clipped) |
| A > 0 | rho < 1 - eps | `surr1` (unclipped is smaller) | `surr1` | `-A * rho` |
| A < 0 | rho inside `[1-eps, 1+eps]` | equal | either | `-A * rho` |
| A < 0 | rho < 1 - eps | `surr2` | `surr2` | **0** (clipped) |
| A < 0 | rho > 1 + eps | `surr1` | `surr1` | `-A * rho` |

The clip only fires when firing it would pull the objective back. That
happens when the policy is already moving in a good direction and has
gone too far. When the policy is moving in the wrong direction (ratio
on the wrong side of 1), the clip does not fire, so the gradient
still pushes the ratio back toward 1.

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

At $\rho = 1$ (the on-policy limit), this simplifies to $-A$, which
recovers the vanilla policy-gradient direction.

At the start of an inner loop, `logprobs_new` should equal
`logprobs_old`, so $\rho \approx 1$. In that first minibatch, the PPO
policy gradient should behave like the vanilla advantage-weighted
log-prob gradient. If it does not, the ratio or the loss sign is
wrong.

For the clipped case, $\mathrm{surr}_2 = (\text{constant w.r.t. }
\theta) \cdot A$. The clip saturates, so changing $\log \pi_\theta$
does not change $\mathrm{surr}_2$. Therefore:

$$
\frac{\partial L}{\partial \log \pi_\theta} = 0
$$

Or in code-style:

    d L / d log pi_theta = 0

That token contributes zero signal to this gradient step. Zero signal
applies only to this gradient step. The rollout sample has already
moved far enough in the advantage-improving direction for the current
inner-loop update. Future rollouts may sample different tokens,
compute different advantages, and produce fresh gradients.

Verify with the **edge test** in Problem 4.5. Set every ratio to a
large value (say `rho = 5`) and `A > 0`. Every token should land in
the "A > 0, rho > 1 + eps, clip active" regime. Autograd must report
exactly zero gradient on those tokens. Any nonzero gradient indicates
a `max` instead of `min`, or a sign error somewhere.

### 2.4 Why this works as a trust region

In practice, on a given update some fraction of tokens hit the clip,
the **clip fraction**, typically 10–30% in a healthy run. Clipped
tokens contribute zero to the gradient, so the effective magnitude of
the update *decreases* as `pi_theta` drifts further from `pi_old`. The
result is a soft self-regulating trust region: small updates move
freely, large ones get throttled automatically.

TRPO, PPO's predecessor, solves a hard constrained optimization at
every step with conjugate-gradient. PPO gets comparable performance
with `min(surr1, surr2)` and a standard optimizer, which is why it
became the default practical choice.

Clip fraction is therefore both a diagnostic and a control signal.
Near 0% suggests updates are tiny or the policy is barely learning.
Near 80% suggests the optimizer is trying to move far beyond the
trust region and the clip is doing most of the work. Healthy PPO
usually lives in the middle.

---

## 3. Value loss (Problem 4.6)

### 3.1 Unclipped form

The value head's job is to regress toward the GAE return:

$$
R_t = A_t + V_{\mathrm{old}}(s_t)
$$

which is computed at rollout time and treated as a constant during
optimization (stop-gradient). Or in code-style:

    R_t = A_t + V_old(s_t)

The simplest value loss is plain MSE:

$$
L_{V,\mathrm{unclipped}} = \frac{1}{2N} \sum_{b,t} m_{b,t} \cdot \bigl(V_\theta(s_t) - R_t\bigr)^2
$$

Or in code-style:

    L_V_unclipped = (1 / (2*N)) * sum over (b, t) of mask[b, t] * (V_theta(s_t) - R_t)^2

The gradient is `mask * (V_theta - R) * dV_theta/dtheta`, the standard
MSE gradient.

The factor of `1/2` is there so the derivative of the square is
clean: derivative of `0.5 * (V - R)^2` with respect to `V` is `V - R`.

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

The value is allowed to move at most `eps_v` away from its snapshot
value `V_old` per update. The reason `max` is used here (rather than
`min` as in the policy case) is that squared error is being
*minimized*, and the larger squared error (the worse of the two
predictions) is pessimistically chosen as the training loss. That
choice prevents `V_theta` from jumping too far in any single update.

The value clip is less discussed than the policy clip but matters in
RLHF because reward scales are learned rather than fixed by an
environment. Early in training, the value head sees noisy returns
whose scale changes as the policy moves. Clipping prevents the value
function from overreacting to one batch and poisoning the next
batch's advantages.

### 3.3 Why clip the value at all

Early in PPO training, the policy is barely moving while the value
head is learning fast from scratch. Without clipping, the value head
can overshoot, predicting return values that do not match the scale
of actual future returns. The TD errors `delta_t` then become garbage,
the policy loss feeds on garbage advantages, and everything
destabilizes.

Clipping `V_theta` to stay within `eps_v` of its previous value keeps
advantages on a stable scale across updates, at the cost of slower
value learning.

This is a deliberate tradeoff. A slightly underfit value function
gives noisier advantages. An unstable value function gives wrong
advantages. PPO usually prefers the first problem because noise
averages out, while systematically wrong advantages push the policy
in bad directions.

In the on-policy limit (when `V_theta = V_old`), both branches of
the max are equal and the clip has no effect. As `V_theta` drifts,
the clip kicks in.

### 3.4 Gradient of the clipped value loss

Piecewise linear in `V_theta`:

- Where `|V_theta - V_old| <= eps_v`, the clip is inactive,
  `V_clipped = V_theta`, and the two branches of the max are equal.
  The gradient is `mask * (V_theta - R)`.
- Where the clip is active:
  - The clipped branch `(V_clipped - R)^2` has no gradient through
    `V_theta` (`V_clipped` is pinned).
  - The unclipped branch `(V_theta - R)^2` has the usual gradient.
  - Whichever branch is picked by `max` determines the gradient.

Gradient-check the whole thing at fp64 in Problem 4.6. Include a
case where `V_theta` is far from both `V_old` and `R` to verify the
gradient matches the selected branch.

### 3.5 Worked example: clipped value loss

Set `eps_v = 0.2`. One token at one position:

    V_old = 0.5
    V_theta = 1.5
    R = 0.7

Compute both branches:

    V_clipped = V_old + clip(V_theta - V_old, -eps_v, +eps_v)
              = 0.5 + clip(1.0, -0.2, +0.2)
              = 0.5 + 0.2 = 0.7

    branch_unclipped = (V_theta - R)^2     = (1.5 - 0.7)^2 = 0.64
    branch_clipped   = (V_clipped - R)^2   = (0.7 - 0.7)^2 = 0.00

    per_tok_loss = 0.5 * max(0.64, 0.00) = 0.32

`max` picks the unclipped branch. Gradient through `V_theta`:

    d(per_tok_loss) / dV_theta = (V_theta - R) = +0.8

After one optimizer step at lr 1.0, `V_theta` moves to `0.7`,
matching `R`. Without the clip, plain MSE would still have moved
`V_theta` to `0.7` after one step on this single example, with the
same direction and magnitude, because the unclipped branch was
selected. The clip only matters when its branch wins the `max`.

Now flip: same `V_old = 0.5`, `R = 0.7`, but `V_theta = 0.55`. Then
`V_clipped = 0.55` (clip inactive), both branches are `(0.55 - 0.7)^2
= 0.0225`, the `max` is the same number, and the gradient is
`(V_theta - R) = -0.15`. Plain MSE behavior near the on-policy limit.

### 3.6 Why the clip helps early in training

Suppose the value head is fresh and outputs `V_old = 0.0` at every
position. The first batch returns `R` values around `5.0` (the
reward model produced larger numbers than the value head expected).
Without clipping, the value gradient pushes every position's
`V_theta` toward `5.0` in one step, which overshoots and feeds
garbage advantages into the next iteration. With `eps_v = 0.2`, no
single step can move `V_theta` more than `0.2` away from `V_old`, so
the value head learns gradually and the policy never sees a single
iteration of severely miscalibrated advantages. The cost is slower
value learning; the benefit is that the policy stops diverging.

---

## 4. Entropy bonus (Problem 4.7)

### 4.1 Purpose

Add the negated entropy of the policy to the loss:

$$
L_{\mathrm{total}} = L_{\mathrm{policy}} + c_v \cdot L_{\mathrm{value}} - c_{\mathrm{ent}} \cdot H(\pi_\theta)
$$

Or in code-style:

    L_total = L_policy + c_v * L_value - c_entropy * H(pi_theta)

The negative sign means a gradient step *increases* entropy, rewarding
the policy for keeping its per-token distributions broad. This
prevents **premature mode collapse**, where the policy becomes almost
deterministic (always picks the argmax) early on and stops exploring
alternatives.

In text, entropy collapse often appears as repetitive phrasing, very
short generic answers, or the same high-reward pattern across many
prompts. The entropy bonus is not a quality objective by itself; it
is a pressure that keeps exploration alive while the reward and KL
terms shape behavior.

In the default config, `c_entropy = 0.0`. If the policy goes
deterministic within a few iterations (entropy dropping near zero,
generations becoming repetitive), bump it to `0.01`.

### 4.2 Masked entropy

For one position, the entropy of the per-token distribution $p$ over
the vocab is:

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

Let $z$ be the logits and $p = \mathrm{softmax}(z)$. Start from the
definition:

$$
H = -\sum_v p_v \log p_v
$$

Differentiating $p_v \log p_v$ with respect to $z_u$ gives $(\partial
p_v / \partial z_u) \cdot (\log p_v + 1)$ (product rule, using
$\partial \log p_v / \partial p_v = 1/p_v$). So:

$$
\frac{\partial H}{\partial z_u}
 = -\sum_v \frac{\partial p_v}{\partial z_u} \cdot \bigl(\log p_v + 1\bigr)
$$

Plug in the softmax derivative $\partial p_v / \partial z_u =
p_v(\delta_{v,u} - p_u)$ from `02-sft.md`:

$$
\frac{\partial H}{\partial z_u}
 = -\sum_v p_v (\delta_{v,u} - p_u) \cdot (\log p_v + 1)
$$

Split $(\delta_{v,u} - p_u)$ into the two pieces. The $\delta_{v,u}$
piece picks out only the $v = u$ term. The $-p_u$ piece factors out of
the sum:

$$
\frac{\partial H}{\partial z_u}
 = - p_u (\log p_u + 1) + p_u \sum_v p_v (\log p_v + 1)
$$

The sum on the right equals $-H + 1$ (since $\sum_v p_v \log p_v = -H$
and $\sum_v p_v = 1$). Substituting:

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

The sign can be checked at the extremes. If one token has probability
near 1, its `log p` is near 0 while `H` is small but positive, so the
gradient of entropy with respect to that dominant logit is negative.
The loss contains `-c_entropy * H`, so gradient descent reduces that
dominant logit, spreading probability mass.

### 4.4 Sanity check

- **Uniform distribution** (`p[v] = 1/V` for all `v`): `log p[v] =
  -log V`, `H = log V`, so `log p + H = 0`. The gradient is zero
  everywhere; entropy is stationary. Uniform is the maximum-entropy
  distribution, so zero gradient at the maximum is the expected
  result.
- **Near-deterministic distribution**: for the wrong classes, `log
  p[v]` is very negative and the gradient is highly nonzero, pointing
  in a direction that smooths the distribution back out.

### 4.5 Test

Gradient-check the entropy at fp64 on a small logits tensor. Apply
the mask and verify that flipping logits *outside* the mask does not
change the computed entropy or its gradient.

### 4.6 Worked example: entropy on a 3-class softmax

Logits `z = (1, 0, -1)`. Softmax:

    exp(1) ≈ 2.718,  exp(0) = 1.000,  exp(-1) ≈ 0.368
    Z = 4.086
    p = (0.665, 0.245, 0.090)

Entropy:

    H = -sum p * log p
      = -(0.665 * (-0.408) + 0.245 * (-1.407) + 0.090 * (-2.408))
      ≈ 0.272 + 0.345 + 0.217
      ≈ 0.834 nats

Gradient `dH/dz = -p * (log p + H)`:

    log p ≈ (-0.408, -1.407, -2.408)
    log p + H ≈ ( 0.426, -0.573, -1.574)
    dH/dz ≈ -p * (log p + H)
          ≈ -(0.665, 0.245, 0.090) * (0.426, -0.573, -1.574)
          ≈ (-0.283, +0.140, +0.142)

The dominant logit (index 0) has `dH/dz < 0`, so increasing it
*decreases* entropy. The two smaller logits have `dH/dz > 0`, so
raising them *increases* entropy by spreading mass. The PPO loss
contains `-c_entropy * H`, so gradient descent on the loss adds
`c_entropy * dH/dz` to each logit's update. With `c_entropy > 0`,
that pushes the dominant logit down and the small logits up,
smoothing the distribution. With `c_entropy = 0` (the default), the
term contributes nothing.

Two sanity-check limits:

- Uniform `p = (1/3, 1/3, 1/3)`: `log p = -log 3` everywhere, `H =
  log 3`, so `log p + H = 0` for every class and `dH/dz = 0`. Uniform
  is the entropy maximum.
- Near-deterministic `p ≈ (0.999, 0.0005, 0.0005)`: `H ≈ 0.008`, the
  small classes have `log p ≈ -7.6`, `log p + H ≈ -7.59`, and `dH/dz`
  for those classes is large positive (about `+0.0038`). Tiny in
  absolute scale because `p` is tiny, but enough to nudge the
  distribution back toward something less peaky.

---

## 5. The full PPO loss

Combine the three pieces:

    L_PPO = L_policy + c_v * L_value - c_entropy * H

Typical coefficients (the default config):

- `c_v = 0.5`: the value loss has half the weight of the policy loss.
- `c_entropy = 0.0` to start; raise to `0.01` if entropy collapses.
- `epsilon = 0.2` for the policy ratio clip.
- `epsilon_v = 0.2` for the value clip.

Backward through the sum, clip the gradient norm at 1.0, step the
optimizer. Four epochs over the rollout batch (Problem 5.3) before
discarding the rollout and starting the next outer iteration.

All three terms share the same logits / backbone in the default
implementation, so their relative coefficients matter. A huge value
coefficient turns PPO into mostly value regression. A huge entropy
coefficient prevents the policy from becoming decisive. A policy LR
that is too high makes the clip fraction spike even when the loss
numbers look finite.

---

## 6. What to commit to `notes/04-ppo-policy.md`

After Problems 4.5, 4.6, 4.7, add:

- Your own derivation of the clipped-surrogate piecewise gradient
  (the table in §2.2, with the case-by-case derivation written out).
- Your own derivation of the entropy gradient (section 4.3 redone by
  hand).
- The numerical result of your edge test: "with every ratio set to 5
  and A > 0, each masked token's gradient was exactly 0.0".
- A note on what coefficients you tried and what you saw happen.
