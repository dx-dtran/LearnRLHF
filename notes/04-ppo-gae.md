# 04 — PPO: policy gradient, advantages, and GAE

## Purpose

Theory packet for Problem 4.4 (and the conceptual backbone for all of
Module 4). This file covers:

1. The policy gradient theorem.
2. Why a baseline is subtracted (variance reduction).
3. The bias–variance tradeoff between one-step TD and full Monte Carlo
   returns.
4. GAE, the recursion that interpolates between those two extremes.
5. How all of this works for per-token rewards on text.

Related notes: `04-ppo-kl.md` (the KL penalty) and `04-ppo-policy.md`
(the PPO clipped surrogate, value loss, entropy). Read those after this
one.

This note connects "we sampled text and assigned rewards" to "we have a
tensor called `advantages` that PPO can optimize." Wrong advantage
estimates push tokens in the wrong direction with confidence, so GAE
gets its own note and its own tests.

For every generated token, PPO needs to decide whether that exact token
should become more likely next time. The advantage answers that.
Positive advantage means the token led to a better outcome than
expected. Negative advantage means the token led to a worse outcome
than expected. GAE turns delayed rewards, value predictions, and masks
into those per-token signals.

---

## 1. Notation for RL on text

The "environment" is one episode of assistant generation. Given a
prompt `s_0` (a sequence of tokens), the policy `pi_theta` generates a
response token by token:

- **State at time `t`**: `s_t = (prompt, a_0, a_1, ..., a_{t-1})`,
  everything the model has seen and produced so far.
- **Action at time `t`**: `a_t`, the next token, sampled from
  `pi_theta(. | s_t)`.
- **Transition**: deterministic. `s_{t+1} = s_t` with `a_t` appended.
- **Reward**: a per-token signal `r_t` constructed from the RM and a KL
  penalty. Defined in `04-ppo-kl.md`. For now treat `r_t` as given.
- **Termination**: the episode ends at time `T` when the policy emits
  `<|im_end|>`, or when it hits the maximum response length.

The objective is the expected total reward over a sampled trajectory:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_t \gamma^t r_t \right]
$$

Or in code-style:

    J(theta) = E_{tau ~ pi_theta} [ sum_t gamma^t * r_t ]

$\gamma$ is a discount factor in $[0, 1]$. For text RL we use $\gamma =
1$: episodes are short, and credit assignment should flow across the
entire response without exponential dampening.

With $\gamma = 1$, later text is not discounted merely because it
appears later in the response. Credit assignment still depends on the
GAE parameter $\lambda$ and the value baseline.

### 1.1 A concrete rollout

Suppose the prompt is:

    s_0 = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"

The model generates one token at a time:

| $t$ | State $s_t$                                    | Action $a_t$ (sampled) | Reward $r_t$            |
|-----|------------------------------------------------|------------------------|-------------------------|
| 0   | prompt                                         | `"The"`                | small KL penalty only   |
| 1   | prompt + `"The"`                               | `" answer"`            | small KL penalty only   |
| 2   | prompt + `"The answer"`                        | `" is"`                | small KL penalty only   |
| 3   | prompt + `"The answer is"`                     | `" 4"`                 | small KL penalty only   |
| 4   | prompt + `"The answer is 4"`                   | `"."`                  | small KL penalty only   |
| 5   | prompt + `"The answer is 4."`                  | `<|im_end|>`           | KL penalty + RM reward  |

Each "state" is the entire token sequence so far. Each "action" is the
next token the policy sampled. Each "reward" is a scalar computed per
token. Most are small KL penalties; the last also carries the terminal
RM reward (the RM's scalar opinion of the whole response).

That is one rollout. An iteration of PPO generates a batch of many such
rollouts in parallel. Every quantity in the rest of this note is
defined on these per-token tuples $(s_t, a_t, r_t)$.

For text, the environment transition is simple: append the sampled
token. The hard part is credit assignment. A good final answer may
depend on many earlier tokens, and a bad answer can be caused by a
single early commitment that made the rest of the response hard to
recover.

---

## 2. The policy gradient theorem

### 2.1 What it says

The gradient of the expected return with respect to the policy
parameters $\theta$ can be written as:

$$
\nabla_\theta J = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \hat G_t \right]
$$

Or in code-style:

    grad J = E_{tau ~ pi_theta} [ sum_t grad log pi_theta(a_t | s_t) * G_hat_t ]

where $\hat G_t$ is any unbiased estimate of "the return from time $t$
onward", that is, any random variable whose expectation equals
$\mathbb{E}\bigl[\sum_{k \ge t} \gamma^{k-t} r_k\bigr]$. The simplest
choice is the Monte Carlo return, the actual cumulative reward observed
from time $t$:

$$
\hat G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k
$$

Or in code-style:

    G_hat_t = sum over k = t..T-1 of gamma^{k-t} * r_k

### 2.2 Where it comes from (sketch)

The full derivation is worth working through on paper at least once.
Sketch:

**Step 1.** Write the objective as a sum (or integral) over
trajectories:

$$
J(\theta) = \sum_\tau p_\theta(\tau) \cdot R(\tau)
$$

where $R(\tau) = \sum_t \gamma^t r_t$ is the total return of the
trajectory.

**Step 2.** Factor the trajectory probability:

$$
p_\theta(\tau) = P(s_0) \prod_t \pi_\theta(a_t \mid s_t) \cdot P(s_{t+1} \mid s_t, a_t)
$$

Only the policy factors depend on $\theta$. The initial state
distribution and the transition dynamics are part of the environment
and are not under the policy's control.

**Step 3.** The **log-derivative trick**. For any positive function
$f(\theta)$:

$$
\nabla_\theta f = f \cdot \nabla_\theta \log f
$$

This follows from the chain rule for $\log$: $\nabla_\theta \log f =
(1/f) \cdot \nabla_\theta f$, multiplied through by $f$. The trick
converts "gradient of a probability" into "probability times gradient
of log-probability", which makes the gradient an expectation that can
be estimated by averaging over samples.

Apply it to $p_\theta(\tau)$:

$$
\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \cdot \nabla_\theta \log p_\theta(\tau)
$$

**Step 4.** Take the gradient of $J$ and push it inside the sum:

$$
\nabla_\theta J = \sum_\tau \nabla_\theta p_\theta(\tau) \cdot R(\tau)
 = \sum_\tau p_\theta(\tau) \cdot \nabla_\theta \log p_\theta(\tau) \cdot R(\tau)
 = \mathbb{E}_\tau \left[ \nabla_\theta \log p_\theta(\tau) \cdot R(\tau) \right]
$$

**Step 5.** Take $\log$ of the factored $p_\theta(\tau)$ and then the
gradient:

$$
\nabla_\theta \log p_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

The $P(s_0)$ and transition terms drop out because they do not depend
on $\theta$.

**Step 6.** Apply causality. Action $a_t$ can only affect rewards from
time $t$ onward, so $R(\tau)$ inside the sum can be replaced with the
return-to-go $\hat G_t$ without changing the expectation.

The result: at each step, the model is nudged to make the action it
took more (or less) likely, weighted by how good the future turned out
to be.

The theorem avoids differentiating through the sampling operation.
Sampling a token is discrete, so ordinary backpropagation cannot pass
through "which token was chosen." The log-derivative trick moves the
gradient onto the log-probability assigned to the sampled token. That
is the core trick behind policy gradients.

### 2.3 What this means intuitively

At each position `t`, for the token `a_t` actually sampled, the model
is nudged to make that token more likely (positive gradient on its
log-probability) when the future went well (`G_hat_t > 0`), and less
likely when it went badly. The size of the nudge is proportional to
how much better or worse than zero the future was.

Monte Carlo returns `G_hat_t` are very noisy. A single trajectory's
cumulative reward depends on many sampled tokens' worth of randomness.
Even with a good policy, the gradient estimate can have huge variance.
The rest of this note is about lowering that variance without
introducing bias.

For language, the variance problem is severe. A response receives one
terminal RM score, but hundreds of sampled token decisions contributed
to that score. If every token receives the same raw terminal return,
the gradient is noisy and blunt. Advantages sharpen that signal.

---

## 3. Baselines and advantages

For any function $b(s_t)$ that depends only on the state (not on the
action that was taken):

$$
\mathbb{E}_{a \sim \pi}\left[ \nabla_\theta \log \pi(a \mid s_t) \cdot b(s_t) \right] = 0
$$

Because $b(s_t)$ does not depend on $a$, it pulls out of the
expectation over $a$. What remains is $\mathbb{E}_a[\nabla \log \pi(a
\mid s_t)]$, which is zero. The one-line proof of that inner identity:

$$
\mathbb{E}_a \bigl[\nabla \log \pi(a \mid s)\bigr]
 = \sum_a \pi(a \mid s) \cdot \frac{\nabla \pi(a \mid s)}{\pi(a \mid s)}
 = \sum_a \nabla \pi(a \mid s)
 = \nabla \sum_a \pi(a \mid s)
 =  \nabla 1  =  0
$$

So any state-dependent $b(s_t)$ can be subtracted from the return
estimate without changing the expected gradient:

$$
\nabla_\theta J = \mathbb{E}\left[ \sum_t \nabla_\theta \log \pi(a_t \mid s_t) \cdot \bigl(\hat G_t - b(s_t)\bigr) \right]
$$

Or in code-style:

    grad J = E[ sum_t grad log pi(a_t | s_t) * (G_hat_t - b(s_t)) ]

Unbiased for any choice of $b$. A good choice of $b$ slashes the
variance of the estimator. The best choice is $b(s_t) = \mathbb{E}[\hat
G_t \mid s_t]$, the expected return starting from $s_t$, since
subtracting it leaves only the action-specific deviation from the
average.

A neural network $V(s_t)$, the **value function**, is trained to
approximate that expectation. The observed return minus this expected
return is the **advantage**:

$$
A_t  =  \hat G_t  -  V(s_t)
$$

The advantage measures how much better (or worse) this action turned
out than expected before taking it. In PPO code, a positive advantage
means "increase the probability of this sampled token"; a negative
advantage means "decrease it." The sign interpretation is a fast
sanity check for the whole RL stack: if the loss makes positive-
advantage tokens less likely, the policy gradient sign is wrong.

The variance-reduced policy gradient:

$$
\nabla_\theta J = \mathbb{E}\left[ \sum_t \nabla_\theta \log \pi(a_t \mid s_t) \cdot A_t \right]
$$

Or in code-style:

    grad J = E[ sum_t grad log pi(a_t | s_t) * A_t ]

Still unbiased, much lower variance in practice. This is what every
modern policy-gradient algorithm uses.

---

## 4. TD error and the bias–variance spectrum

Before defining GAE, one more building block: the **one-step TD
error**.

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

Or in code-style:

    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

The TD error is "the reward I actually got, plus what the value
function says is left to earn from the next state, minus what I
expected from this state." It is the part of the return at time $t$
that was not predicted by the value function. (TD stands for "temporal
difference".)

The TD error functions as a surprise signal for the value function.
Positive $\delta_t$ means the step turned out better than the value
estimate predicted. Negative $\delta_t$ means it turned out worse. GAE
accumulates these surprises backward through time.

Advantage estimators built from TD errors can use varying numbers of
steps:

### 4.1 One-step advantage

Use one TD error:

$$
A_t^{(1)} = \delta_t
$$

Or in code-style:

    A_t^(1) = delta_t

- **Low variance**: depends on only one reward sample.
- **High bias**: relies heavily on `V(s_{t+1})` being accurate. If `V`
  is wrong, so is this advantage.

### 4.2 Monte Carlo advantage

Sum rewards all the way to the episode end:

$$
A_t^{(\infty)} = \sum_{k=t}^{T-1} \gamma^{k-t} r_k - V(s_t)
$$

Or in code-style:

    A_t^(infinity) = sum over k = t..T-1 of gamma^{k-t} * r_k  -  V(s_t)

- **Zero bias**: does not trust `V` at all except as the baseline at
  time `t`.
- **High variance**: the tail of rewards is noisy.

### 4.3 n-step in between

$n$ real rewards followed by a bootstrap with $V$:

$$
A_t^{(n)}
 = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n}) - V(s_t)
$$

An equivalent form in terms of TD errors:

$$
A_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k \delta_{t+k}
$$

Or in code-style:

    A_t^(n) = r_t + gamma*r_{t+1} + ... + gamma^{n-1}*r_{t+n-1} + gamma^n*V(s_{t+n}) - V(s_t)
            = sum over k = 0..n-1 of gamma^k * delta_{t+k}

(The second line is an algebraic identity: expand the definition of
$\delta$ and the intermediate $V$ terms telescope. The $+\gamma
V(s_{t+1})$ from $\delta_t$ cancels the $-V(s_{t+1})$ from
$\delta_{t+1}$, and so on.)

As `n` grows, bias falls and variance rises. GAE provides a single
knob that interpolates between these choices.

This is the bias–variance tradeoff in value-based credit assignment.
Trusting the value function more inherits its bias. Trusting sampled
returns more inherits their noise. GAE gives a dial for choosing
between those errors.

---

## 5. Generalized Advantage Estimation (Schulman et al. 2016)

### 5.1 Definition

GAE is an exponentially weighted average of the n-step advantages for
all $n \ge 1$. In terms of TD errors:

$$
A_t^{\mathrm{GAE}} = \sum_{k=0}^{\infty} (\gamma \lambda)^k \delta_{t+k}
$$

Or in code-style:

    A_t^GAE = sum over k = 0, 1, 2, ... of (gamma * lambda)^k * delta_{t+k}

The knob is $\lambda$ in $[0, 1]$:

- `lambda = 0`: GAE collapses to one-step TD, `A_t = delta_t`. Low
  variance, high bias.
- `lambda = 1`: GAE collapses to Monte Carlo. No bias, high variance.
- `lambda` in between: smooth interpolation. The standard choice for
  text RL is `lambda = 0.95`.

### 5.2 Deriving the recursion

Split off the first term of the sum, then factor out one $(\gamma
\lambda)$ from the rest:

$$
A_t^{\mathrm{GAE}}
 = \delta_t + \sum_{k=1}^{\infty} (\gamma\lambda)^k \delta_{t+k}
 = \delta_t + (\gamma\lambda) \sum_{k=0}^{\infty} (\gamma\lambda)^k \delta_{t+1+k}
 = \delta_t + (\gamma\lambda) A_{t+1}^{\mathrm{GAE}}
$$

Or in code-style:

    A_t = delta_t + sum over k = 1, 2, ... of (gamma*lambda)^k * delta_{t+k}
        = delta_t + (gamma*lambda) * sum over k = 0, 1, ... of (gamma*lambda)^k * delta_{t+1+k}
        = delta_t + (gamma*lambda) * A_{t+1}

So the GAE advantage satisfies the one-line recursion:

$$
A_t = \delta_t + \gamma\lambda A_{t+1}
$$

Boundary condition: at the terminal time $T$, the episode is over, so
$V(s_T) = 0$ by convention and $A_T = 0$.

The recursion runs *backwards* through time, from `T-1` down to `0`,
and that is how it is computed in code. `A[t]` depends on `A[t+1]`,
so the future advantage must already be known. Vectorized versions
exist, but the Python backward loop is clearer and fast enough for
the small response lengths in this repo.

### 5.3 Algorithm

```
A[T]     = 0
V[T]     = 0              # by convention
for t = T-1, T-2, ..., 0:
    delta_t = r[t] + gamma * V[t+1] * nonterm[t+1] - V[t]
    A[t]    = delta_t + gamma * lambda * A[t+1] * nonterm[t+1]
```

The `nonterm[t+1]` factor handles variable-length sequences in a
batched loop. It is 1 if position `t+1` is still part of the real
episode and 0 if it is padding or beyond the end. When `nonterm = 0`,
the bootstrap from the future is zeroed out.

Without this mask, one row's padding becomes fake future reward, and
advantages leak across the artificial tail after EOS. The risk is
especially concrete in batched text generation because different rows
finish at different times.

### 5.4 Returns as value targets

After computing advantages, the value head needs a regression target:

$$
R_t  =  A_t  +  V_t
$$

Or in code-style:

    R_t = A_t + V_t

This is sometimes called the "bootstrapped Monte Carlo return". If
`A_t` estimates "how much better the return was than expected" and
`V_t` estimates "what we expected", their sum estimates the actual
return.

**Important.** When training the value head, `R_t` is treated as a
stop-gradient *constant*, even though it was computed from `V_t`. If
not, the value head chases its own tail: it tries to make `V_t` match
something that itself depends on `V_t`, and nothing trains.

In practice: compute the advantages and returns under
`torch.no_grad()` (or detach the values first). The `ppo_core.gae`
function can stay pure; the `train_ppo.py` driver wraps it correctly.

If the stop-gradient rule is forgotten, the value loss can partially
cancel itself through the target. The code still runs, but the
optimization problem becomes "move predictions and labels together",
which weakens or destroys the value training signal.

### 5.5 Unit test (Problem 4.4)

Hand-computed 3-step example. Set `T = 3`, `values = [0, 0, 0]`,
`rewards = [1, 0, 0]`, `mask = [1, 1, 1]`, `gamma = 1`, `lambda = 1`.

Walking the recursion backwards:

- `delta_2 = 0 + 0 - 0 = 0`, so `A_2 = 0`.
- `delta_1 = 0 + 0 - 0 = 0`, so `A_1 = 0 + 1 * 0 = 0`.
- `delta_0 = 1 + 0 - 0 = 1`, so `A_0 = 1 + 1 * 0 = 1`.

Result: `advantages = [1, 0, 0]`, `returns = [1, 0, 0]`.

Add a second test with a pad token in the middle. The implementation
should handle the `nonterm` mask correctly, zeroing out the bootstrap
at the pad position.

### 5.6 Worked example: GAE with a non-zero value baseline

The all-zero example above hides what GAE actually does, since the
value function is silent. A more realistic case: `T = 3`, `gamma = 1`,
`lambda = 0.95`, `nonterm = [1, 1, 1, 0]` (the position after the last
real step is terminal). Per-token rewards and value estimates:

    rewards = [0.0, 0.0, 1.0]
    values  = [0.5, 0.3, 0.0]                 # V(s_T) = 0 by convention

Step backward through `delta_t = r_t + gamma * V_{t+1} - V_t` and
`A_t = delta_t + gamma * lambda * A_{t+1}`.

    t = 2:
        delta_2 = 1.0 + 1*0 - 0.0   = +1.0
        A_2     = 1.0 + 0.95 * 0     = +1.0

    t = 1:
        delta_1 = 0.0 + 1*0.0 - 0.3 = -0.3
        A_1     = -0.3 + 0.95 * 1.0  = +0.65

    t = 0:
        delta_0 = 0.0 + 1*0.3 - 0.5 = -0.2
        A_0     = -0.2 + 0.95 * 0.65 = +0.4175

So `advantages = [0.4175, 0.65, 1.0]` and `returns = A + V = [0.9175,
0.95, 1.0]`.

The structure: the value head expected to receive about `0.5` total
return at `t=0` and `0.3` at `t=1`, but the actual outcome was a single
unit of reward at `t=2`. GAE attributes most of the surprise to the
last step (where the reward arrives) and propagates a smaller share
back to earlier steps, discounted by `(gamma * lambda)`.

### 5.7 Worked example: lambda = 0 collapse to one-step TD

Same rewards `[0, 0, 1]`, same `values = [0.5, 0.3, 0.0]`, same `gamma
= 1`. With `lambda = 0`, the recursion `A_t = delta_t + gamma * lambda
* A_{t+1}` loses the future term entirely:

    t = 2: A_2 = delta_2 = +1.0
    t = 1: A_1 = delta_1 = -0.3
    t = 0: A_0 = delta_0 = -0.2

So `advantages = [-0.2, -0.3, 1.0]`. This is exactly the per-step TD
error. Earlier steps now get *negative* advantages because the value
head's optimistic predictions were wrong about *those* steps in
isolation, even though the trajectory eventually paid off. Low variance
(each advantage is one TD error), high bias (the future is summarized
entirely by `V`).

### 5.8 Worked example: lambda = 1 collapse to Monte Carlo

Same setup, `lambda = 1`. The recursion becomes a plain backward sum
of TD errors:

    t = 2: A_2 = +1.0
    t = 1: A_1 = -0.3 + 1.0 = +0.7
    t = 0: A_0 = -0.2 + 0.7 = +0.5

So `advantages = [0.5, 0.7, 1.0]`. Algebraically, `A_t = (sum_{k=t..T-1}
r_k) - V(s_t)` here: Monte Carlo return minus the baseline. Higher
variance because each advantage carries the full reward tail, but no
systematic bias from `V`'s bootstraps.

Compare the three rows:

```
lambda     A_0     A_1     A_2
0.00     -0.20   -0.30   +1.00       # one-step TD
0.95     +0.42   +0.65   +1.00       # standard PPO setting
1.00     +0.50   +0.70   +1.00       # Monte Carlo
```

`lambda` slides between the all-bias and all-variance corners. PPO's
preferred `0.95` lives close to the Monte Carlo end but inherits
enough of `V` to dampen variance.

### 5.9 Worked example: padded batch

Two rollouts of different lengths, both padded to length 5. The mask
`nonterm[t]` is `1` while inside the real episode and `0` afterwards.

    row 0: actual length 3, nonterm = [1, 1, 1, 0, 0]
    row 1: actual length 5, nonterm = [1, 1, 1, 1, 1]

Per-row rewards and values:

    row 0: rewards = [0, 0, 1, 0, 0]   values = [0.4, 0.2, 0.0, 0.0, 0.0]
    row 1: rewards = [0, 0, 0, 0, 1]   values = [0.6, 0.4, 0.3, 0.1, 0.0]

For row 0, the recursion only attends to positions 0..2. At `t=3`,
`nonterm = 0`, so `delta_3 = r_3 + gamma * V_4 * 0 - V_3 = 0`, and
`A_3 = 0 + gamma*lambda * A_4 * 0 = 0`. Same at `t=4`. Without the
`nonterm` factor on the recursion's bootstrap term, advantages from
positions 3 and 4 (which are just padding) would leak into position
2's computation. The mask zeroes them, so row 0's GAE advantages match
what would be computed on a length-3 sequence in isolation.

This is the bug that Problem 4.4's pad-in-the-middle test catches. If
the `nonterm` factor is forgotten on the bootstrap, both rows still
produce numbers, but row 0's advantages will be wrong in a way that
depends on the (arbitrary) padding values.

---

## 6. Per-token rewards for text

In classical RL, the environment hands out a reward at every step. For
text the per-token reward is constructed:

$$
r_t = -\beta \cdot \mathrm{KL}_t(\pi_\theta \| \pi_{\mathrm{ref}}) + r_{\mathrm{RM}}(y) \cdot \mathbf{1}[ t = T_{\mathrm{last}} ]
$$

Or in code-style:

    r_t = -beta * KL_t(pi_theta || pi_ref)  +  r_RM(y) * (t == last_response_token)

Two pieces:

- A small per-token KL penalty that punishes the policy for drifting
  from the frozen reference (SFT) model at every step of the response.
- A terminal RM reward, delivered as a single scalar at the last real
  token of the response.

See `04-ppo-kl.md` for the KL penalty details: why it is there, what
estimator is used, and how it is computed.

The RM reward is computed after the full response exists, while the
KL cost is available at every generated token. GAE combines these two
time scales: local KL penalties and a delayed terminal preference
score.

Two consequences:

1. **The per-token reward is not the RM score alone.** The KL penalty
   is baked in at the per-token level, so the policy learns to trade
   off reward seeking against staying close to the SFT distribution
   along the whole response, not just at the end.
2. **GAE propagates the terminal RM reward backward through the
   response.** The advantage at an intermediate token reflects how
   much the terminal reward exceeded expectation, minus the KL cost
   accumulated since that point. GAE's exponential weighting decides
   how much credit each intermediate token gets for the final outcome.

---

## 7. Advantage normalization (Problem 4.8)

Before feeding advantages into the PPO policy loss, normalize them
across valid tokens to mean 0 and std 1:

$$
\mu = \frac{\sum_{b,t} m_{b,t} A_{b,t}}{\sum_{b,t} m_{b,t}},
\qquad
\sigma^2 = \frac{\sum_{b,t} m_{b,t} (A_{b,t} - \mu)^2}{\sum_{b,t} m_{b,t}},
\qquad
\tilde A = \frac{A - \mu}{\sqrt{\sigma^2} + \varepsilon}
$$

Or in code-style:

    mean = (sum over (b, t) of mask[b, t] * A[b, t])  /  sum(mask)
    var  = (sum over (b, t) of mask[b, t] * (A[b, t] - mean)^2)  /  sum(mask)
    A_tilde = (A - mean) / (sqrt(var) + epsilon)

The PPO clip range (typically `epsilon = 0.2`) was chosen for
advantages on roughly unit scale. If the RM happens to output rewards
on the scale of 10 or 100, advantages are correspondingly huge, every
step hits the clip, and no learning happens. Normalization makes the
effective learning rate independent of the absolute scale the RM
happens to have learned.

Compute the mean and variance only over positions where the mask is 1
(real response tokens). Padding tokens are garbage data, and including
them in the stats pollutes the normalization. The unit test in Problem
4.8 inserts huge garbage values at masked positions and asserts that
the normalized output's mean and std are unchanged.

After normalization, masked positions can contain any value as long
as the loss later ignores them. Zeroing them out after normalization
is often cleaner for inspection, but the mathematical requirement is
only that they do not influence the mean, variance, or PPO loss.

---

## 8. What to commit to `notes/04-ppo-gae.md`

After finishing Problems 4.4 and 4.8, add:

- Your own derivation of the GAE recursion (section 5.2 above, redone
  by hand).
- The numeric output of your unit tests, including the
  pad-in-the-middle case.
- A one-line explanation of what `lambda` does.
