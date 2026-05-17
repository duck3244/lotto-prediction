"""예측 서비스 — 활성 모델 + 현재 데이터로 한 사이클 예측을 수행."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from analysis import analyze_patterns, ensemble_prediction
from constants import LOTTO_COLUMNS
from data_loader import get_latest_sequence
from model import predict_next_numbers
from utils import set_global_seeds, suggest_balanced_numbers

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    lstm: list[int]
    ensemble: list[int]
    additional_sets: list[list[int]]
    sequence_length: int
    seed: int


def make_prediction(
    *,
    model: Any,
    scaler: Any,
    df: pd.DataFrame,
    sequence_length: int,
    seed: int,
    num_sets: int = 3,
) -> PredictionResult:
    """현재 활성 모델로 다음 회차 예측 + 앙상블 + 보조 추천 세트 생성."""
    set_global_seeds(seed)

    latest_sequence = get_latest_sequence(df, sequence_length, scaler)
    lstm = predict_next_numbers(model, latest_sequence, scaler)

    frequencies, _, _ = analyze_patterns(df)
    ensemble = ensemble_prediction(df, lstm, frequencies, sequence_length)

    recent_numbers: set[int] = set()
    for i in range(min(5, len(df))):
        for col in LOTTO_COLUMNS:
            recent_numbers.add(int(df.iloc[i][col]))

    additional = (
        suggest_balanced_numbers(frequencies, recent_numbers, num_sets) if num_sets > 0 else []
    )

    return PredictionResult(
        lstm=[int(x) for x in lstm],
        ensemble=[int(x) for x in ensemble],
        additional_sets=[[int(x) for x in s] for s in additional],
        sequence_length=int(sequence_length),
        seed=int(seed),
    )
