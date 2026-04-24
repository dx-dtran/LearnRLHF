# 04 — PPO: clipped surrogate, value loss, entropy

## Purpose

Theory for Problems 4.5, 4.6, and 4.7. Derive and reason about the three terms in the
PPO optimization objective:

$$
\mathcal{L}_\text{PPO}(\theta) = \mathcal{L}_\pi(\theta) + c_v \cdot \mathcal{L}_V(\theta) - c_\text{ent} \cdot H(\theta).
$$

By the end you should be able to:

1. State the PPO clipped policy loss and derive its piecewise gradient.
2. Explain why the clipped ratio provides a "trust region" without actually solving
   one.
3. Derive the clipped value loss and explain why the clipping helps early in
   training.
4. Derive the gradient of the masked entropy term.

---

## 1. The setup: importance-sampled policy gradient

Recall from `04-ppo-gae.md`: the vanilla policy gradient is

$$
\nabla_\theta J = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_t \nabla \log \pi_\theta(a_t \mid s_t) \cdot A_t\right].
$$

**Problem**: to compute this gradient we need rollouts from the *current* policy
$\pi_\theta$. If we want to reuse a batch of rollouts for multiple gradient steps (to
amortize the cost of generation), the rollouts quickly become "off-policy" and the
expectation is wrong.

**Fix**: importance sampling. Rollouts come from a *snapshot* policy $\pi_\text{old}$
(frozen at the start of each outer iteration). We rewrite the objective as

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\text{old}}\!\left[\sum_t \frac{\pi_\theta(a_t \mid s_t)}{\pi_\text{old}(a_t \mid s_t)} \cdot A_t\right],
$$

where the importance ratio

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_\text{old}(a_t \mid s_t)}
= \exp\bigl(\log \pi_\theta(a_t \mid s_t) - \log \pi_\text{old}(a_t \mid s_t)\bigr)
$$

accounts for the distribution shift. Under mild conditions $\nabla_\theta$ on this
expression equals $\nabla_\theta J$, as long as $\pi_\theta \approx \pi_\text{old}$.

**The catch**: if $\pi_\theta$ drifts far from $\pi_\text{old}$, the ratios can be
huge and the gradient estimator blows up with variance. This is the classic failure
of importance sampling. We need a way to reuse the batch but prevent the policy
from moving too far in one update.

---

## 2. The clipped surrogate (Schulman et al. 2017)

### 2.1 Definition

$$
\mathcal{L}_t^\text{clip}(\theta) = -\min\bigl(\rho_t(\theta) \cdot A_t, \; \text{clip}(\rho_t(\theta), 1 - \varepsilon, 1 + \varepsilon) \cdot A_t\bigr),
$$

$$
\mathcal{L}_\pi(\theta) = \frac{1}{N} \sum_{b, t} m_{b, t} \cdot \mathcal{L}^\text{clip}_{b, t}(\theta).
$$

where $N$ is the count of real (masked-in) response tokens in the batch and
$\varepsilon = 0.2$ is the clip range. Per token:

- $\text{surr}_1 = \rho A$ is the importance-sampled policy gradient objective.
- $\text{surr}_2 = \text{clip}(\rho, 1-\varepsilon, 1+\varepsilon) \cdot A$ is the same
  with $\rho$ pinned to $[1-\varepsilon, 1+\varepsilon]$.
- We take the **min** inside the loss — i.e. the **pessimistic** one — so the policy
  is not rewarded for running up the ratio beyond the clip in a direction that
  increases the objective. *But* it is still allowed to pull the ratio back toward 1
  when that would decrease the objective.

### 2.2 What "pessimistic" means, concretely

There are four cases based on the sign of $A_t$ and whether $\rho$ is inside the clip:

| $A_t$ sign | $\rho$ position            | surr$_1$ vs surr$_2$        | min picks | Gradient w.r.t. $\log \pi$ |
|------------|----------------------------|-----------------------------|-----------|----------------------------|
| $A > 0$    | $\rho \in [1-\varepsilon, 1+\varepsilon]$ | equal                | surr$_1$  | $-A \cdot \rho$            |
| $A > 0$    | $\rho > 1 + \varepsilon$   | surr$_1$ bigger (= $\rho A$) | surr$_2$  | **zero** (clipped)         |
| $A > 0$    | $\rho < 1 - \varepsilon$   | surr$_1$ smaller            | surr$_1$  | $-A \cdot \rho$ (not clipped)|
| $A < 0$    | $\rho \in [1-\varepsilon, 1+\varepsilon]$ | equal                | surr$_1$  | $-A \cdot \rho$            |
| $A < 0$    | $\rho < 1 - \varepsilon$   | surr$_1$ less negative      | surr$_2$  | **zero** (clipped)         |
| $A < 0$    | $\rho > 1 + \varepsilon$   | surr$_1$ more negative      | surr$_1$  | $-A \cdot \rho$ (not clipped)|

The pattern: **clipping only fires when it would drag the objective back** (i.e.
the policy is already moving in a good direction and has gone "too far"). It *does
not* fire when the ratio is on the wrong side — in those cases the unclipped branch
provides gradient signal back toward the clip range.

### 2.3 Deriving the piecewise gradient (do this by hand)

For the unclipped case ($\text{surr}_1$ picked), the per-token loss is
$\ell = -\rho A$. Using $\rho = \exp(\log \pi - \log \pi_\text{old})$:

$$
\frac{\partial \rho}{\partial \log \pi} = \rho,
\qquad
\frac{\partial \ell}{\partial \log \pi} = -A \cdot \rho.
$$

Equivalently, $\partial \ell / \partial \log \pi = -A$ when $\rho = 1$ — recovering the
vanilla policy gradient in the on-policy limit.

For the clipped case ($\text{surr}_2$ picked *and* it doesn't depend on $\theta$ in
that regime — i.e. the clip is saturated): the argument of the clip has no gradient,
so $\partial / \partial \log \pi = 0$. The token's gradient contribution is literally
zero.

You verify this with an **edge test** in Problem 4.5: set all ratios to a large value
($\rho = 5$, say) and $A > 0$. Every token should be in the "$A > 0$, $\rho > 1+\varepsilon$,
clip picks surr$_2$" regime. Autograd must report exactly zero gradient on those
tokens. If it reports nonzero, your implementation is using `max` instead of `min`
or has a sign error.

### 2.4 Why this works as a trust region substitute

The "clip fraction" (fraction of tokens where the clip fires) is typically 10–30% of
tokens per iteration. When the clip fires, that token contributes zero signal to the
update, so the effective gradient magnitude decreases as $\theta$ drifts farther from
$\theta_\text{old}$. This acts like a soft trust region: small updates move freely,
large updates get throttled automatically.

Contrast with TRPO (the predecessor), which solves a constrained optimization at every
step with a conjugate-gradient solver — PPO gets comparable performance with a simple
`min(surr1, surr2)` and a standard optimizer. That's why everyone uses PPO.

---

## 3. Value loss (Problem 4.6)

### 3.1 Unclipped form

The value head's job is to regress onto the GAE return $R_t = A_t + V_t^\text{old}$
(which we computed at rollout time and treat as a constant):

$$
\mathcal{L}_V^\text{unclipped}(\theta) = \frac{1}{2} \sum_{b, t} m_{b, t} \bigl(V_\theta(s_{b, t}) - R_{b, t}\bigr)^2 \Big/ N.
$$

Straightforward MSE; gradient is $m \cdot (V - R) \cdot \partial V/\partial \theta$.

### 3.2 Clipped form

$$
V_\theta^\text{clipped}(s_t) = V^\text{old}_t + \text{clip}\bigl(V_\theta(s_t) - V^\text{old}_t,\; -\varepsilon_v,\; +\varepsilon_v\bigr),
$$

$$
\mathcal{L}_V(\theta) = \frac{1}{2} \sum_{b, t} m_{b, t} \cdot \max\!\bigl((V_\theta - R)^2,\; (V_\theta^\text{clipped} - R)^2\bigr) \Big/ N.
$$

Same "pessimistic" idea as the policy clip: whichever of $V_\theta$ or $V_\theta^\text{clipped}$
is **farther** from the target $R$ determines the squared error. Why **max** (not
min) here? Because squared error is a loss we want to *minimize*, and we want to
pessimistically compute how bad the prediction is — i.e. use the larger of the two
squared errors. That prevents $V_\theta$ from jumping too far in one update.

### 3.3 Why clip the value

Early in PPO training, the policy is barely changing while the value head is
catching up fast. Without clipping, the value can overshoot — predicting $V$ values
that don't match the scale of future returns — and this wrecks the advantage
estimates used by the policy loss (since the TD errors $\delta_t$ become garbage).
Clipping $V$ per-update keeps the value estimate close to its own previous estimate,
so advantages stay on a stable scale for the policy to consume.

In the on-policy limit ($V_\theta = V^\text{old}$), both branches give the same value
and the clip has no effect. As $V_\theta$ drifts, the clip kicks in.

### 3.4 Gradient

The gradient is piecewise linear in $V_\theta$:

- Where $|V_\theta - V^\text{old}| \le \varepsilon_v$: the clip is inactive, and
  $(V_\theta - R)^2$ and $(V_\theta^\text{clipped} - R)^2$ are equal. Gradient =
  $m \cdot (V_\theta - R)$.
- Where the clip is active: the clipped branch's gradient through
  $V_\theta$ is zero (it was clipped). If the unclipped branch has the larger squared
  error (max picks it), gradient = $m \cdot (V_\theta - R)$. Otherwise the max picks
  the clipped branch, whose gradient w.r.t. $\theta$ is zero — no signal.

The test in Problem 4.6: gradient-check the whole thing at fp64, and add a case where
$V_\theta$ is far from $V^\text{old}$ and far from $R$ — check the gradient equals what
you'd compute for the selected branch.

---

## 4. Entropy bonus (Problem 4.7)

### 4.1 Purpose

Adding an entropy term to the loss with negative sign

$$
-c_\text{ent} \cdot H(\pi_\theta) = -c_\text{ent} \cdot \mathbb{E}\!\left[-\sum_v \pi_\theta(v \mid s) \log \pi_\theta(v \mid s)\right]
$$

rewards policies that keep distributional "width". This prevents premature mode
collapse — where the policy becomes deterministic (always picks the argmax) too
early and stops exploring.

In our config: `entropy_coef = 0.0` by default. Start without; add a small amount
(`0.01`) if you see the policy go deterministic within a few iterations (entropy
dropping near zero, generations becoming repetitive).

### 4.2 Masked entropy over response tokens

$$
H(\pi_\theta) = \frac{1}{N} \sum_{b, t} m_{b, t} \cdot H\bigl(\pi_\theta(\cdot \mid s_{b, t})\bigr),
$$

where per-token entropy is

$$
H_t = -\sum_{v=1}^{V} \pi_\theta(v \mid s_t) \log \pi_\theta(v \mid s_t).
$$

### 4.3 Gradient of $H$ with respect to logits (derive this)

Let $z \in \mathbb{R}^V$ be logits, $p = \text{softmax}(z)$. Using
$\partial p_v / \partial z_u = p_v(\delta_{v, u} - p_u)$ from the SFT note:

$$
\frac{\partial H}{\partial z_u}
= -\sum_v \frac{\partial p_v}{\partial z_u} (\log p_v + 1)
= -\sum_v p_v(\delta_{v, u} - p_u)(\log p_v + 1).
$$

Expand the two terms in the parenthesis and simplify:

$$
\frac{\partial H}{\partial z_u} = -p_u(\log p_u + 1) + p_u \sum_v p_v(\log p_v + 1)
= -p_u \log p_u - p_u + p_u(- H + 1)
= -p_u (\log p_u + H).
$$

So in vector form:

$$
\boxed{\; \frac{\partial H}{\partial z} = -\, p \odot \bigl(\log p + H\bigr). \;}
$$

### 4.4 Sanity check

At the uniform distribution $p_v = 1/V$: $\log p_v = -\log V$, $H = \log V$, and
$\log p_v + H = -\log V + \log V = 0$. So $\partial H/\partial z = 0$ — entropy is
stationary (maximized) at uniform. Good.

At a near-deterministic $p$ (most mass on one class): $\log p_v$ is very negative for
all but one $v$, and the gradient is highly nonzero in a direction that smooths the
distribution. Also good.

### 4.5 Test

Gradient-check at fp64 on a small logits tensor. Apply the mask, verify that flipping
logits outside the mask doesn't change the entropy value or its gradient.

---

## 5. The full PPO loss

Putting the three pieces together:

$$
\mathcal{L}_\text{PPO} = \mathcal{L}_\pi + c_v \cdot \mathcal{L}_V - c_\text{ent} \cdot H.
$$

Typical coefficients (our config):

- $c_v = 0.5$: value loss has half the weight of policy loss.
- $c_\text{ent} = 0.0$ (start), up to $0.01$ if needed.
- $\varepsilon = 0.2$ for the policy ratio clip.
- $\varepsilon_v = 0.2$ for the value clip.

Backward through the sum, clip gradients to norm 1.0, step the optimizer. Four epochs
over the rollout batch (Problem 5.3) before throwing away the rollout and starting the
next outer iteration.

---

## 6. What to commit to `notes/04-ppo-policy.md`

After finishing Problems 4.5, 4.6, 4.7, append:

- Your derivation of the clipped-surrogate piecewise gradient (the table in §2.2,
  but with the algebra filled in).
- Your derivation of the entropy gradient (redo §4.3 on paper).
- The numerical result of your edge test: "with all ratios at 5 and A > 0, every
  masked token's gradient is 0.0".
- A note on what coefficients you tried and what happened.
