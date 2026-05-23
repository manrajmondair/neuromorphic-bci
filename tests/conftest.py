"""Shared pytest fixtures.

Tests are intentionally tiny — they cover the invariants of the
data-ridge-baseline side of the pipeline. They run against mock data so
they don't need the real NLB MC_RTT download.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `from src.* import ...` works
# when pytest is invoked from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from src.utils.mock_data import make_mock_processed_data  # noqa: E402


@pytest.fixture(scope="module")
def mock_data():
    """A small mock processed-data dict matching docs/data_interface.md."""
    return make_mock_processed_data(num_bins=400, num_neurons=24, seed=0)
