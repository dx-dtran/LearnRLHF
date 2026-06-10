# solutions/

Complete reference implementations of every `[FILL]` blank in the scaffold files.

Try each blank yourself first — the blanks are the course. Look here when you've
been stuck for ~20 minutes, or to compare after your tests pass.

Two ways to use these:

- Run the whole test suite against the solutions (sanity-check a confusing test, or
  verify your environment):

      LEARNRLHF_SOLUTIONS=1 pytest tests/

- Run a training script directly from this directory (uses solution code for the
  blanks, repo-root code for everything else):

      python solutions/train_sft.py

Only files containing blanks are mirrored here (`model.py`, `tokenizer.py`,
`train_sft.py`, `train_rm.py`, `ppo_core.py`). Everything else — `data_hh.py`,
`train_ppo.py`, `eval.py`, `config.py`, `grad_check.py` — is already complete at the
repo root.
