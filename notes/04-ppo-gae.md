# 04 — PPO: policy gradient, advantages, and GAE

## Purpose

Theory for Problem 4.4 (and the conceptual backbone for all of Module 4). This file
covers:

1. The policy gradient theorem.
2. Why we subtract a baseline — variance reduction.
3. The bias–variance spectrum between one-step TD and Monte Carlo returns.
4. The GAE recursion that linearly combines them, and how to derive it from scratch.
5. How to use GAE on *per-token* rewards for language generation.

The closely related KL penalty and PPO policy loss have their own note files
(`04-ppo-kl.md` and `04-ppo-policy.md`). Read those after this one.

---

## 1. Notation for RL on text

Our "environment" is a single episode of assistant generation. For a given prompt $s_0$
(which is a sequence of tokens), the policy $\pi_\theta$ rolls out a response token by
token:

- State at time $t$: $s_t = (\text{prompt}, a_0, a_1, \dots, a_{t-1})$ — everything the
  model has seen so far.
- Action at time $t$: $a_t$, the next token sampled from $\pi_\theta(\cdot \mid s_t)$.
- Transition: deterministic. $s_{t+1} = s_t \circ a_t$ (concatenation).
- Reward: given by a **reward signal** we construct from the RM and a KL penalty —
  we'll define this in `04-ppo-kl.md`. For now treat $r_t$ as given per token.
- Episode ends at $t = T$ when the policy emits `<|im_end|>` or hits the max response
  length.

The objective:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T-1} \gamma^t r_t\right],
$$

where $\tau$ is a full trajectory and $\gamma \in [0, 1]$ is the discount. For text we
use $\gamma = 1$: trajectories are short and we want credit assignment across the
whole response.

---

## 2. The policy gradient theorem

### 2.1 Statement

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \hat G_t\right],
$$

where $\hat G_t$ is any unbiased estimate of $\mathbb{E}[\sum_{k \ge t} \gamma^{k-t} r_k]$,
the return from $t$ onward. The simplest choice is the Monte Carlo return itself:

$$
\hat G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k.
$$

### 2.2 Where it comes from

Sketch of the derivation (do it in full on paper at least once):

- $J(\theta) = \sum_\tau p_\theta(\tau) R(\tau)$ where $R(\tau)$ is total return.
- $p_\theta(\tau) = P(s_0) \prod_t \pi_\theta(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t)$.
  Only the $\pi_\theta$ terms depend on $\theta$.
- $\nabla \log p_\theta(\tau) = \sum_t \nabla \log \pi_\theta(a_t \mid s_t)$.
- Apply the log-derivative trick: $\nabla J = \mathbb{E}[\nabla \log p_\theta \cdot R]$.
- Causality: action $a_t$ can only affect rewards at times $\ge t$, so you can replace
  $R$ with $\hat G_t$ inside the sum.

### 2.3 What it means in practice

At every position $t$, for every token $a_t$ we sampled, we push its log-probability
**up** in proportion to how good the resulting future was ($\hat G_t > 0$) or **down**
if it was bad ($\hat G_t < 0$).

The policy gradient is intuitive and correct, but $\hat G_t$ (the Monte Carlo return)
has very high variance — one trajectory's sum is noisy, and trajectories are
expensive (each one requires a full rollout). Reducing variance without introducing
bias is the whole game from here.

---

## 3. The baseline: advantage estimation

Observation: for any function $b(s_t)$ that depends only on the state (not the action),

$$
\mathbb{E}_{a_t \sim \pi}\!\left[\nabla \log \pi(a_t \mid s_t) \cdot b(s_t)\right] = 0.
$$

(Proof: $b$ pulls out of the expectation over $a_t$; what remains is
$\mathbb{E}_a[\nabla \log \pi] = \nabla \mathbb{E}_a[1] = 0$.)

So subtracting $b(s_t)$ from $\hat G_t$ does not change the expected gradient. But it
can drastically reduce variance — especially if $b(s_t)$ is close to
$\mathbb{E}[\hat G_t \mid s_t]$. The "advantage":

$$
A_t = \hat G_t - V(s_t),
$$

where $V(s_t)$ is our *value estimate* — a learned function (the value head) that
approximates $\mathbb{E}[\hat G_t \mid s_t]$. Intuitively: the advantage is "how much
better did this action turn out than what I expected before I took it?" Subtracting
the baseline removes the bulk of trajectory-level noise (e.g. "this prompt is just an
easy one") and keeps only the action-specific signal.

Variance-reduced policy gradient:

$$
\nabla_\theta J = \mathbb{E}\!\left[\sum_t \nabla \log \pi_\theta(a_t \mid s_t) \cdot A_t\right].
$$

This is still unbiased as long as $V$ is a function of state only.

---

## 4. TD error and the bias–variance spectrum

Define the **one-step TD error** at time $t$ as:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).
$$

This is "the part of the return I didn't expect": the actual reward plus the next
state's value minus the current state's value.

Two extreme advantage estimators:

### 4.1 One-step TD ($\lambda = 0$)

$$
A_t^{(0)} = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).
$$

- **Low variance** — depends on a single reward sample.
- **High bias** — inherits any error in $V(s_{t+1})$. If $V$ is a bad estimate, so is
  $A^{(0)}$.

### 4.2 Monte Carlo return ($\lambda = 1$)

$$
A_t^{(\infty)} = \sum_{k=t}^{T-1} \gamma^{k-t} r_k - V(s_t)
= \sum_{k=t}^{T-1} \gamma^{k-t}\, \delta_k \cdot \text{(plus telescoping )}.
$$

(See derivation below.)

- **Zero bias** (doesn't use $V$ anywhere in the future; only as the baseline at
  time $t$).
- **High variance** — sums rewards over the whole tail.

### 4.3 n-step returns

For any $n$:

$$
A_t^{(n)} = r_t + \gamma r_{t+1} + \dots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n}) - V(s_t)
= \sum_{k=0}^{n-1} \gamma^k \delta_{t+k}.
$$

Mid-spectrum: bias falls off with $n$, variance grows with $n$. GAE lets us interpolate
between all of these with a single knob $\lambda$.

---

## 5. Generalized Advantage Estimation (Schulman et al. 2016)

### 5.1 Definition

GAE is an exponentially weighted average of the $A_t^{(n)}$ for all $n \ge 1$:

$$
A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{k=0}^{\infty} (\gamma \lambda)^k \, \delta_{t+k}.
$$

- $\lambda = 0$: GAE collapses to the one-step TD error $\delta_t$.
- $\lambda = 1$: GAE collapses to the Monte Carlo advantage (no bias, all variance).
- $\lambda \in (0, 1)$: interpolate. Typical choice for text RL: $\lambda = 0.95$.

### 5.2 Deriving the recursion (do this by hand)

Split the sum at $k = 0$:

$$
A_t = \delta_t + \sum_{k=1}^{\infty} (\gamma \lambda)^k \delta_{t+k}
    = \delta_t + (\gamma \lambda)\sum_{k=0}^{\infty}(\gamma \lambda)^k \delta_{t+1+k}
    = \delta_t + \gamma \lambda \, A_{t+1}.
$$

So

$$
\boxed{\; A_t = \delta_t + \gamma \lambda\, A_{t+1}. \;}
$$

Boundary condition: at the terminal time $T$, the episode ends, so $V(s_T) = 0$ by
convention and $A_T = 0$.

### 5.3 Algorithm

```
V_T = 0
A_T = 0
for t = T-1, T-2, ..., 0:
    delta_t = r_t + gamma * V_{t+1} * nonterm_{t+1} - V_t
    A_t     = delta_t + gamma * lam * A_{t+1} * nonterm_{t+1}
```

where `nonterm_{t+1} = mask[t+1]` (1 if position $t+1$ is real and the episode is not
done there, 0 if it's padding or a post-EOS step). The `nonterm` factor is how you
handle variable-length sequences in a batched loop.

### 5.4 Returns as value targets

After computing advantages, compute the value regression target:

$$
R_t = A_t + V_t.
$$

This is the "bootstrapped Monte Carlo return" that's consistent with $A_t$: if $A_t$
estimates the deviation from $V_t$ and $V_t$ estimates $\mathbb{E}[G_t]$, then
$A_t + V_t$ estimates $G_t$ itself. The value head is trained to regress $V_t$ to
$R_t$ (see `04-ppo-policy.md`).

**Important:** $R_t$ is computed from $V_t$, but we treat $R_t$ as a
**stop-gradient constant** inside the value loss. Otherwise the value head would be
trying to make $V_t$ match something that depends on $V_t$ — a trivial and useless
fixed point. In practice: detach the values before the GAE computation, or call GAE
under `torch.no_grad()`. The `ppo_core.gae` function stays pure; the `train_ppo.py`
driver is responsible for the no-grad context.

### 5.5 Unit test (from Problem 4.4)

Hand-computed 3-step example: $T = 3$, `values = [0, 0, 0]`, `rewards = [1, 0, 0]`,
`mask = [1, 1, 1]`, $\gamma = 1$, $\lambda = 1$.

- $\delta_2 = r_2 + \gamma V_3 - V_2 = 0 + 0 - 0 = 0$, $A_2 = 0$.
- $\delta_1 = r_1 + \gamma V_2 - V_1 = 0 + 0 - 0 = 0$, $A_1 = 0 + 1 \cdot 0 = 0$.
- $\delta_0 = r_0 + \gamma V_1 - V_0 = 1 + 0 - 0 = 1$, $A_0 = 1 + 1 \cdot 0 = 1$.

So `advantages = [1, 0, 0]`, `returns = [1, 0, 0]`. This matches the comment in
`ppo_core.gae`. Add a second test with a pad token in the middle — your implementation
should treat the pad position's advantage as zero.

---

## 6. Per-token rewards for text

In classical RL, $r_t$ is an environmental reward at each step. For text, we
**construct** the per-token reward signal:

$$
r_t = -\beta \cdot \text{KL}_t(\pi_\theta \| \pi_\text{ref}) + r_\text{RM}(y) \cdot \mathbf{1}_{t = T-1}.
$$

- A small **per-token KL penalty** that punishes the policy for moving too far from
  the reference (SFT) model at every token.
- A **terminal RM reward**, delivered as a single scalar at the last real token of the
  response.

See `04-ppo-kl.md` for why and what the $\text{KL}_t$ estimator looks like.

Two consequences:

1. **The "reward" in GAE is not the RM score alone** — the KL penalty is baked in at
   the per-token level, so the policy learns to trade off reward-seeking against
   staying close to the SFT distribution *along the whole response*, not just at the end.
2. **GAE propagates the terminal RM reward backward through the response.** $A_t$ at
   an intermediate token reflects how much the terminal reward exceeded expectation,
   minus the KL cost accumulated since. GAE's exponential weighting decides how much
   credit each intermediate token gets.

---

## 7. Advantage normalization (Problem 4.8)

Before feeding advantages into the PPO policy loss, normalize them across the *valid
tokens* in the batch to zero mean and unit std:

$$
\tilde A_t = \frac{A_t - \mu}{\sigma + \epsilon}, \qquad
\mu = \frac{\sum_{b, t} m_{b, t} A_{b, t}}{\sum_{b, t} m_{b, t}}, \qquad
\sigma^2 = \frac{\sum_{b, t} m_{b, t} (A_{b, t} - \mu)^2}{\sum_{b, t} m_{b, t}}.
$$

Why: keeps the effective learning rate scale-invariant to whatever absolute scale the
RM is on. The PPO clip range $\varepsilon = 0.2$ assumes advantages of roughly unit
scale; without normalization, a large-magnitude RM means every step hits the clip and
no learning happens.

**Crucially** compute $\mu$ and $\sigma$ using only the mask's 1-positions. Pad tokens
are garbage data — they must not pollute the stats. The test in Problem 4.8 inserts
huge garbage values at masked positions and checks the normalized stats are unchanged.

---

## 8. What to commit to `notes/04-ppo-gae.md`

After finishing Problem 4.4 and 4.8, append:

- Your own derivation of the GAE recursion (the 5.2 step).
- The numeric output of your unit tests, including the pad-in-the-middle case.
- A one-line explanation of what $\lambda$ does (you should be able to say this cold).
