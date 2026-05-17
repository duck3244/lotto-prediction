"""순수 헬퍼 함수 테스트 (TF 사이드이펙트 없음).

``utils`` 가 ``tensorflow`` 를 import 하므로 첫 컬렉션 시 TF 로드 비용이 한 번 발생한다.
그 외 본 테스트들은 GPU/모델 없이 즉시 수행된다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from utils import (
    compare_predictions,
    data_hash_matches,
    suggest_balanced_numbers,
    validate_lotto_numbers,
    _sha256_file,
)


# --- validate_lotto_numbers --------------------------------------------------

class TestValidateLottoNumbers:
    def test_valid_set(self):
        ok, msg = validate_lotto_numbers([1, 5, 12, 23, 34, 45])
        assert ok is True

    def test_wrong_count(self):
        ok, msg = validate_lotto_numbers([1, 2, 3, 4, 5])
        assert ok is False
        assert "정확히" in msg

    @pytest.mark.parametrize("nums", [[0, 5, 12, 23, 34, 45], [1, 5, 12, 23, 34, 46]])
    def test_out_of_range(self, nums):
        ok, msg = validate_lotto_numbers(nums)
        assert ok is False
        assert "유효하지 않습니다" in msg

    def test_duplicates(self):
        ok, msg = validate_lotto_numbers([1, 1, 12, 23, 34, 45])
        assert ok is False
        assert "중복" in msg


# --- compare_predictions -----------------------------------------------------

class TestComparePredictions:
    def test_exact_match(self):
        count, matched = compare_predictions([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])
        assert count == 6
        assert matched == [1, 2, 3, 4, 5, 6]

    def test_partial_match(self):
        count, matched = compare_predictions([1, 2, 3, 4, 5, 6], [1, 2, 9, 10, 11, 12])
        assert count == 2
        assert matched == [1, 2]

    def test_no_match(self):
        count, matched = compare_predictions([1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12])
        assert count == 0
        assert matched == []

    def test_empty_inputs(self):
        assert compare_predictions([], [1, 2, 3]) == (0, [])
        assert compare_predictions([1, 2, 3], []) == (0, [])


# --- _sha256_file / data_hash_matches ---------------------------------------

class TestDataHash:
    def test_sha256_file_stable(self, tmp_path: Path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"hello world\n")
        h1 = _sha256_file(f)
        h2 = _sha256_file(f)
        assert h1 == h2
        assert len(h1) == 64

    def test_sha256_changes_on_content_change(self, tmp_path: Path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"a")
        h1 = _sha256_file(f)
        f.write_bytes(b"b")
        h2 = _sha256_file(f)
        assert h1 != h2

    def test_data_hash_matches_true(self, tmp_path: Path):
        f = tmp_path / "data.xlsx"
        f.write_bytes(b"dummy bytes")
        meta = {"data_sha256": _sha256_file(f)}
        assert data_hash_matches(meta, f) is True

    def test_data_hash_matches_false_on_change(self, tmp_path: Path):
        f = tmp_path / "data.xlsx"
        f.write_bytes(b"v1")
        meta = {"data_sha256": _sha256_file(f)}
        f.write_bytes(b"v2")
        assert data_hash_matches(meta, f) is False

    def test_data_hash_matches_missing_meta(self, tmp_path: Path):
        f = tmp_path / "data.xlsx"
        f.write_bytes(b"v1")
        assert data_hash_matches({}, f) is False
        assert data_hash_matches({"data_sha256": None}, f) is False

    def test_data_hash_matches_missing_file(self, tmp_path: Path):
        meta = {"data_sha256": "a" * 64}
        assert data_hash_matches(meta, tmp_path / "nope.xlsx") is False


# --- suggest_balanced_numbers ------------------------------------------------

class TestSuggestBalancedNumbers:
    def test_returns_valid_sets(self):
        np.random.seed(0)
        frequencies = {i: max(1, 50 - abs(23 - i)) for i in range(1, 46)}
        recent_numbers = {3, 14, 25, 36, 41}
        sets = suggest_balanced_numbers(frequencies, recent_numbers, num_to_generate=4)
        assert len(sets) == 4
        for s in sets:
            assert len(s) == 6
            assert len(set(s)) == 6  # 중복 없음
            for n in s:
                assert 1 <= n <= 45
