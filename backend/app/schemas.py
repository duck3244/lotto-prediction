"""Pydantic 응답/요청 모델.

한국어 컬럼명(``회차``, ``번호1``…)은 응답 경계에서 영어 키로 변환한다 — 프론트
코드 가독성과 i18n 미래 대비.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# --- 공통 ---

class HealthResponse(BaseModel):
    status: str
    tensorflow_version: Optional[str] = None
    gpu_available: bool = False


# --- 회차 / 통계 ---

class DrawRow(BaseModel):
    draw_no: int
    numbers: list[int]


class RecentDrawsResponse(BaseModel):
    total_draws: int
    rows: list[DrawRow]


class FrequencyEntry(BaseModel):
    number: int
    count: int


class StatsResponse(BaseModel):
    total_draws: int
    frequencies: list[FrequencyEntry]
    odd_even: dict[str, int]
    range_distribution: dict[str, int]


# --- 모델 / 번들 ---

class BundleSummary(BaseModel):
    name: str
    timestamp: Optional[str] = None
    sequence_length: Optional[int] = None
    seed: Optional[int] = None
    data_sha256: Optional[str] = None
    tensorflow_version: Optional[str] = None
    sklearn_version: Optional[str] = None
    is_active: bool = False
    data_hash_match: bool = False


class ModelsResponse(BaseModel):
    bundles: list[BundleSummary]
    active_name: Optional[str] = None


class ActivateBundleRequest(BaseModel):
    name: str = Field(..., description="활성화할 번들 디렉토리 이름 (예: bundle_20260517_120000)")


class ActivateBundleResponse(BaseModel):
    active_name: str
    data_hash_match: bool
    message: str


# --- 예측 ---

class PredictRequest(BaseModel):
    sequence_length: Optional[int] = Field(
        default=None,
        description="None 이면 활성 번들의 메타에 기록된 sequence_length 를 사용.",
    )
    seed: int = 42
    num_sets: int = Field(default=3, ge=0, le=20)


class PredictResponse(BaseModel):
    lstm: list[int]
    ensemble: list[int]
    additional_sets: list[list[int]]
    active_bundle: str
    data_hash_match: bool
    sequence_length: int
    seed: int


# --- 학습 ---

class TrainRequest(BaseModel):
    epochs: int = Field(default=300, ge=1, le=2000)
    batch_size: int = Field(default=64, ge=1, le=1024)
    sequence_length: int = Field(default=10, ge=2, le=50)
    seed: int = 42
    auto_activate: bool = True


class TrainJobParams(BaseModel):
    epochs: int
    batch_size: int
    sequence_length: int
    seed: int
    auto_activate: bool


class TrainJob(BaseModel):
    job_id: str
    status: str
    epoch: int
    total_epochs: int
    best_val_loss: Optional[float] = None
    last_loss: Optional[float] = None
    last_val_loss: Optional[float] = None
    error: Optional[str] = None
    bundle_name: Optional[str] = None
    submitted_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    params: TrainJobParams


class TrainJobListResponse(BaseModel):
    current_job_id: Optional[str] = None
    jobs: list[TrainJob]
