"""회차 데이터 / 통계 라우터 — `/api/draws/*`."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, Query

from analysis import analyze_patterns
from app.config import settings
from app.deps import get_data_store
from app.schemas import (
    DrawRow,
    FrequencyEntry,
    RecentDrawsResponse,
    StatsResponse,
)
from app.services.data_store import DataStore
from constants import DRAW_COLUMN, LOTTO_COLUMNS

router = APIRouter(prefix="/draws", tags=["draws"])


@router.get("/recent", response_model=RecentDrawsResponse)
def recent_draws(
    limit: int = Query(default=settings.default_recent_limit, ge=1, le=settings.max_recent_limit),
    data_store: DataStore = Depends(get_data_store),
) -> RecentDrawsResponse:
    df = data_store.get()
    head = df.head(limit)
    rows = [
        DrawRow(
            draw_no=int(row[DRAW_COLUMN]),
            numbers=[int(row[c]) for c in LOTTO_COLUMNS],
        )
        for _, row in head.iterrows()
    ]
    return RecentDrawsResponse(total_draws=int(len(df)), rows=rows)


@router.get("/stats", response_model=StatsResponse)
def stats(data_store: DataStore = Depends(get_data_store)) -> StatsResponse:
    df = data_store.get()
    frequencies, odd_even, range_patterns = analyze_patterns(df)

    return StatsResponse(
        total_draws=int(len(df)),
        frequencies=[
            FrequencyEntry(number=int(n), count=int(c))
            for n, c in sorted(frequencies.items())
        ],
        odd_even={k: int(v) for k, v in odd_even.items()},
        range_distribution={k: int(v) for k, v in range_patterns.items()},
    )
