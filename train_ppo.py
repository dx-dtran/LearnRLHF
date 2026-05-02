"""
train_ppo.py — PPO training loop.

Module 5. Glues the ppo_core.py building blocks into an actual RLHF run.

Four models live here (see CLAUDE.md §3):

    policy       : trainable, init from sft.pt
    value        : ScalarHead on TOP of the policy's final hidden states. We share the
                   backbone with the policy (cheaper); the only extra trainable params
                   are the n_embd -> 1 head. If you see value-head instability, swap to
                   a SEPARATE backbone for value (more memory, more stable).
    reference    : FROZEN copy of sft.pt. Used to compute the KL penalty.
    reward_model : FROZEN, loaded from rm.pt. Used to score full (prompt+response).

Each PPO iteration:
    1. ROLLOUT (no-grad everything): sample prompts, generate responses from policy,
       record (logprobs_old, values_old), compute ref logprobs, compute RM reward,
       shape per-token rewards, compute GAE.
    2. OPTIMIZE: K epochs × minibatches over that rollout. Recompute logprobs and
       values with GRAD enabled; combine PPO policy loss + value loss + entropy bonus;
       backward, clip, step.
    3. LOG: reward, KL (k3), policy loss, value loss, entropy, clip frac, tokens/sec.

Memory: all 4 models live in bf16. Ref and RM are frozen, so they do not keep
activations at full precision; call them under `torch.no_grad()` and autocast.
Policy and value accumulate activations. If memory is tight, use gradient
checkpointing on the policy's blocks by wrapping each block's forward in
`torch.utils.checkpoint.checkpoint`.
"""

# =====================================================================================
# Problem 5.1 — Model layout, memory map, smoke test
# =====================================================================================
# Script-level sketch:
#
#   def build_models(cfg):
#       policy  = GPT(cfg).cuda().to(bf16)
#       value_h = ScalarHead(cfg.n_embd).cuda().to(bf16)
#       ref     = GPT(cfg).cuda().to(bf16)          ; ref.requires_grad_(False); ref.eval()
#       rm      = RewardModel(cfg).cuda().to(bf16)  ; rm.requires_grad_(False); rm.eval()
#       load weights: policy <- sft.pt, ref <- sft.pt, rm <- rm.pt
#       return policy, value_h, ref, rm
#
#   after one dummy forward, print torch.cuda.max_memory_allocated() / 1e9 GB.
#
#   TODO(5.1): implement build_models + a main that prints memory.


# =====================================================================================
# Problem 5.2 — Rollout phase
# =====================================================================================
# def rollout(policy, value_h, ref, rm, prompt_batch, cfg):
#     with torch.no_grad(), autocast(bf16):
#         full_ids, response_ids, logprobs_old, values_old, response_mask = \
#             generate_with_logprobs(policy, value_h, prompt_batch["prompt_ids"],
#                                    prompt_batch["prompt_mask"],
#                                    cfg.response_max_len, cfg.temperature, ...)
#         ref_logits  = ref(full_ids, ...)                    # (B, T, V)
#         ref_logprobs = gather_logprobs(ref_logits[:, T_p-1:-1, :], response_ids)
#         rm_reward    = rm(full_ids, full_mask, last_idx)    # (B,)
#         kl_t         = kl_k1(logprobs_old, ref_logprobs)
#         per_tok_r    = shape_reward(rm_reward, kl_t, response_mask, cfg.kl_coef)
#         adv, ret     = gae(per_tok_r, values_old, response_mask, cfg.gamma, cfg.gae_lambda)
#         adv          = normalize_advantages(adv, response_mask)
#     return dict(full_ids=, response_ids=, logprobs_old=, values_old=, advantages=adv,
#                 returns=ret, response_mask=, rm_reward=, kl_t=)
#
# TODO(5.2): implement.


# =====================================================================================
# Problem 5.3 — Optimize phase (inner loop)
# =====================================================================================
# def optimize(policy, value_h, optimizer, rollout, cfg):
#     stats = defaultdict(list)
#     B = rollout["full_ids"].size(0)
#     for epoch in range(cfg.ppo_epochs):
#         for mb_idx in minibatches(B, cfg.minibatch_size):
#             # slice all rollout tensors by mb_idx
#             with autocast(bf16):
#                 hidden = policy.forward_hidden(full_ids_mb, mask_mb)
#                 logits = hidden @ policy.wte.weight.T
#                 values_new = value_h(hidden).squeeze(-1)
#                 # slice to response region
#                 logprobs_new = gather_logprobs(logits[:, T_p-1:-1, :], response_ids_mb)
#                 values_new   = values_new[:, T_p-1:-1]
#                 Lpi = ppo_policy_loss(logprobs_new, logprobs_old_mb, adv_mb, mask_mb, cfg.clip_eps)
#                 Lv  = value_loss(values_new, values_old_mb, returns_mb, mask_mb, cfg.value_clip_eps)
#                 H   = masked_entropy(logits[:, T_p-1:-1, :], mask_mb)
#                 loss = Lpi + cfg.value_coef * Lv - cfg.entropy_coef * H
#             loss.backward()
#             clip_grad_norm_(params, cfg.grad_clip)
#             optimizer.step(); optimizer.zero_grad()
#             stats[...].append(...)
#     return {k: mean(v) for k,v in stats.items()}
#
# TODO(5.3): implement. The response starts at T_p - 1 in logits because logits[i]
# predicts token i+1, so logits[T_p - 1] predicts the first response token.


# =====================================================================================
# Problem 5.4 — Logging
# =====================================================================================
# TODO(5.4):
#   - write stats to a CSV at training time
#   - every N iters, render matplotlib plots of reward, kl, policy_loss, value_loss
#   - print tokens/sec = (B * response_len) / iter_time


# =====================================================================================
# Problem 5.5 — Larger sizes
# =====================================================================================
# After implementing GPTConfig.from_name in config.py:
#
#   def test_model_sizes_smoke():
#       for name in ["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
#           cfg = GPTConfig.from_name(name)
#           m = GPT(cfg).cuda().to(bf16)
#           x = torch.randint(0, cfg.vocab_size, (1, 64), device="cuda")
#           logits = m(x); logits.sum().backward()
#           print(name, torch.cuda.max_memory_allocated() / 1e9)
#           del m; torch.cuda.empty_cache()
#
# TODO(5.5): implement as a `if __name__ == "__main__" and "--smoke" in sys.argv` branch
# or a separate tiny script. Log peak memory to notes/05-ppo.md.


def train_ppo():
    """
    TODO(5.1-5.4): wire everything together:

        cfg = PPOConfig()
        policy, value_h, ref, rm = build_models(GPTConfig(), cfg)
        tokenizer = ...
        train_prompts = PromptDataset(...)
        optim = build_optimizer(...)  # include value_h params
        for it in range(cfg.num_iters):
            batch = next(prompt_iterator)
            ro = rollout(policy, value_h, ref, rm, batch, cfg)
            stats = optimize(policy, value_h, optim, ro, cfg)
            log(stats, it); maybe_save(it)
    """
    raise NotImplementedError("TODO(5.1-5.4): train_ppo")


if __name__ == "__main__":
    train_ppo()