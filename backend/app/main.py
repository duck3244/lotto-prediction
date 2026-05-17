"""FastAPI 애플리케이션 진입점.

- ``lifespan`` 에서 ``DataStore`` / ``ModelStore`` 단일 인스턴스를 만들고,
  마지막 활성 번들이 있으면 복원한다(실패해도 앱은 기동된다).
- CORS 는 ``settings.cors_origins`` 만 허용.
- 모든 라우터는 ``/api`` 프리픽스로 마운트.

본 모듈은 ``run_api.py`` 또는 ``uvicorn app.main:app`` 로 기동된다. **반드시**
``env_setup.set_deterministic_env()`` 가 TF 임포트 이전에 실행되어야 한다 —
``run_api.py`` 가 그 책임을 진다.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import draws as draws_router
from app.routers import health as health_router
from app.routers import models as models_router
from app.routers import predict as predict_router
from app.routers import training as training_router
from app.services.data_store import DataStore
from app.services.model_store import ModelStore
from app.services.training import TrainingService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI 초기화 시작")

    data_store = DataStore()
    model_store = ModelStore()

    # 데이터는 첫 요청에서 지연 로드(import 사이클을 짧게). 활성 번들은 미리 복원.
    try:
        restored = model_store.restore_from_disk()
        if restored:
            logger.info("활성 번들 복원 완료: %s", restored)
        else:
            logger.info("활성 번들 없음 — /api/models/active 로 선택 필요")
    except Exception as e:
        logger.warning("활성 번들 복원 중 예외: %s", e)

    training_service = TrainingService(data_store=data_store, model_store=model_store)

    app.state.data_store = data_store
    app.state.model_store = model_store
    app.state.training_service = training_service

    yield

    logger.info("FastAPI 종료")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Lotto Prediction API",
        version="0.1.0",
        description="LSTM + 앙상블 로또 번호 예측 (단일 사용자 MVP).",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(health_router.router, prefix="/api")
    app.include_router(draws_router.router, prefix="/api")
    app.include_router(models_router.router, prefix="/api")
    app.include_router(predict_router.router, prefix="/api")
    app.include_router(training_router.router, prefix="/api")

    return app


app = create_app()
