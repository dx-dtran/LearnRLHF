"""
tests/conftest.py — path setup + course-friendly test behavior.

Two things happen here:

1. The repo root goes on sys.path so flat-file imports work. If the env var
   LEARNRLHF_SOLUTIONS=1 is set, solutions/ goes FIRST, so every test runs against
   the reference implementation instead of your fills. That is how the solutions are
   kept verified; you can also use it to check whether a failure is your code or
   the test:   LEARNRLHF_SOLUTIONS=1 pytest tests/

2. A blank you have not filled in yet raises NotImplementedError; those tests show
   as SKIPPED, not failed. A failing test therefore always means implemented-but-
   wrong.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLUTIONS = os.path.join(ROOT, "solutions")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if os.environ.get("LEARNRLHF_SOLUTIONS") == "1":
    # Preload the reference implementations under their bare module names so every
    # `import model` (etc.) anywhere in the suite hits solutions/, regardless of
    # whatever pytest prepends to sys.path later.
    import importlib.util

    for _name in ("model", "tokenizer", "train_sft", "train_rm", "ppo_core"):
        _spec = importlib.util.spec_from_file_location(
            _name, os.path.join(SOLUTIONS, _name + ".py")
        )
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_name] = _mod
        _spec.loader.exec_module(_mod)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow and/or requires network")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("-m"):
        skip_slow = pytest.mark.skip(reason="slow; run with `pytest -m slow`")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except NotImplementedError as e:
        pytest.skip(f"blank not filled in yet: {e}")
