# LearnRLHF

From-scratch InstructGPT-style RLHF on GPT-2, single 24GB GPU, pure PyTorch.

This README is the **on-ramp**: setup, the daily workflow, how to know you're on track.
The full curriculum, difficulty estimates, derivations you need to produce, and the
spec for each of the ~34 problems live in [`CLAUDE.md`](./CLAUDE.md) — read that once up
front and then keep it open in a split.

---

## 1. Setup (once)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/test_grad_check.py -q     # should pass: 3 tests
```

If that last command shows `3 passed`, your grad-check harness works and you can trust
every other gradient check in the repo.

You'll also need to download the Anthropic HH-RLHF dataset — that happens inside
`data_hh.download_hh()` which you implement in Problem 0.2.

---

## 2. How the curriculum is structured

- **6 modules**, numbered 0–6. Each module has several **problems** numbered `M.P`.
- Every problem has **three artifacts**:
  1. a `# TODO(M.P):` block in a skeleton file telling you exactly what to implement,
  2. a derivation you write in `notes/0M-*.md` (for anything with a nontrivial backward pass),
  3. a matching test in `tests/` that must pass before you move on.

Estimated total focused time: **~18–22 hours**. See `CLAUDE.md §2` for per-phase breakdown.

---

## 3. The daily loop (do this for EVERY problem)

For problem `M.P`:

1. **Read.** Open `CLAUDE.md §8` and find problem `M.P`. Read the `# TODO(M.P):` block
   in the skeleton file — the docstring above it states the math and what the tests
   check.
2. **Derive.** If the problem has a backward pass (SFT, BT, PPO, KL, GAE, value,
   entropy), write the derivation in `notes/0M-<topic>.md` *before* typing code.
   Paraphrase nothing — if you can't derive it on paper, you don't understand it yet.
3. **Implement** the `TODO`.
4. **Test.** Run only the test(s) for this problem. Don't run the whole suite:
   ```bash
   pytest tests/test_grad_ppo.py::test_ppo_policy_loss_grad -q
   ```
5. **Green means green.** Gradient checks in particular — if rel_err is `1e-3` on a
   loss that should be `1e-6`, your analytic grad is *almost* right, which is worse
   than completely wrong. Find the off-by-one / sign / mask / shift before continuing.
6. **Commit** with `git commit -m "M.P: <one line>"`. Small commits make bisecting
   trivial when a later module breaks an earlier one.

---

## 4. Order you do them in

You MUST go in order — later problems import earlier ones. The authoritative list is
`CLAUDE.md §8`. Starting here:

```
0.1 clean slate (done for you — old code is in old/)
0.2 HH download + parse          → tests/test_data.py
1.1–1.8 GPT-2 from scratch       → tests/test_model.py
2.1–2.5 SFT                      → tests/test_tokenizer.py, test_grad_sft.py
3.1–3.5 Reward model             → tests/test_grad_rm.py
4.1–4.8 PPO building blocks      → tests/test_grad_ppo.py
5.1–5.5 PPO training loop        → (mostly smoke tests + real training)
6.1–6.3 Final eval               → notes/06-eval.md
```

When in doubt: run `pytest tests/ -q` and work on the FIRST failing test (tests are
named so their filenames sort roughly in curriculum order).

---

## 5. What "done" looks like

Per-problem: the named test passes.

Per-module: all tests in its file pass AND you've written the notes derivation.

Whole repo: the definition-of-done checklist at the bottom of `CLAUDE.md §10`. At a
minimum:
- full `pytest tests/ -q` is green
- `python eval.py --models base,sft,rlhf --n 20` produces a markdown table where the
  RLHF column is visibly better at instruction-following on most rows
- you can derive, on a whiteboard, the backward pass of every loss in the repo

---

## 6. Debugging tips that will save you hours

- **Gradient checks lie only when you lie to them.** If a grad check passes but
  training does nothing, the bug is in the masking / shift / reduction, not the
  gradient. Look at `.shape` and `.sum()` of every mask.
- **PPO "trains" while being silently broken.** Always watch all six logs
  simultaneously: reward, KL (k3), policy loss, value loss, entropy, clip fraction. If
  reward goes up while KL explodes, you're reward-hacking — lower LR or raise β.
- **When confused about alignment** in PPO (logits[t] vs response[t]), write a 3-token
  example on paper and verify indices BEFORE changing code.
- **`old/` is your oracle** for cases where you want to see how someone else solved a
  piece (imperfectly — that's why we're rewriting). Don't copy, but do diff against it
  when stuck.

---

## 7. Where to write what

- **Code:** the skeleton file the `TODO(M.P)` lives in. No new files, no new folders.
- **Derivations:** `notes/0M-<topic>.md`. One per module minimum. Pencil-and-paper
  quality. Include at least one worked numeric example per loss.
- **Logs & plots:** `runs/<timestamp>/` (create it when training starts).
- **Checkpoints:** `sft.pt`, `rm.pt`, `rlhf.pt` at repo root (paths configurable in
  `config.py`).

---

## 8. A few norms (also in `CLAUDE.md §11`)

- No `transformers.Trainer`, no `trl`, no `accelerate`, no `peft`, no `deepspeed`.
  `transformers` is only allowed in `load_gpt2_from_hf` for the one-time weight load.
- No `ignore_index=-100`. Always an explicit `loss_mask` tensor, multiplied in.
- `nn.Module` yes; dataclasses yes; anything more abstract — no.
- Comments should explain *why*, not *what*. Code explains what.

---

Good luck. Read `CLAUDE.md` once, come back here when you sit down to work, and treat
every failing test as the next step — not as a setback.