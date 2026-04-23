"""
tests/conftest.py — make the repo root importable without __init__.py files, and
register the `slow` marker used by test_model.py::test_hf_parity.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: test is slow and/or requires network")


def pytest_collection_modifyitems(config, items):
    import pytest
    if not config.getoption("-m"):
        skip_slow = pytest.mark.skip(reason="slow; run with `-m slow`")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)