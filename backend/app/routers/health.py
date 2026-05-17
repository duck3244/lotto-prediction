"""헬스 체크 — `/api/health`."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    # tf import 는 비용이 크지만 본 라우터 호출 이전에 이미 lifespan 에서 끝난다.
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    return HealthResponse(
        status="ok",
        tensorflow_version=tf.__version__,
        gpu_available=bool(gpus),
    )
