"""``evaluate._random_baseline_matches`` 단위 테스트.

전체 분포가 아니라 단발 호출의 정상 작동(범위/개수)만 검증한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluate import _random_baseline_matches


def test_returns_count_in_valid_range():
    rng = np.random.default_rng(42)
    actual = [3, 7, 14, 21, 32, 41]
    for _ in range(20):
        c = _random_baseline_matches(actual, rng)
        assert 0 <= c <= 6


def test_deterministic_with_same_rng_seed():
    actual = [3, 7, 14, 21, 32, 41]
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    seq1 = [_random_baseline_matches(actual, rng1) for _ in range(10)]
    seq2 = [_random_baseline_matches(actual, rng2) for _ in range(10)]
    assert seq1 == seq2
