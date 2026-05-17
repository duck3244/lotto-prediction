"""예측 라우터 — `POST /api/predict`."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.config import settings
from app.deps import get_data_store, get_model_store
from app.routers.training import get_training_service
from app.schemas import PredictRequest, PredictResponse
from app.services.data_store import DataStore
from app.services.model_store import ModelStore
from app.services.prediction import make_prediction
from app.services.training import TrainingService
from utils import data_hash_matches

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(
    body: PredictRequest,
    data_store: DataStore = Depends(get_data_store),
    model_store: ModelStore = Depends(get_model_store),
    training_service: TrainingService = Depends(get_training_service),
) -> PredictResponse:
    if training_service.is_running():
        raise HTTPException(
            status_code=409,
            detail="학습 잡이 진행 중입니다. 완료된 후 다시 시도하세요.",
        )

    model, scaler, meta, active_name = model_store.snapshot()
    if model is None or scaler is None or meta is None or not active_name:
        raise HTTPException(
            status_code=400,
            detail="활성 모델이 없습니다. /api/models/active 로 먼저 번들을 선택하세요.",
        )

    sequence_length = (
        body.sequence_length
        if body.sequence_length is not None
        else int(meta.get("sequence_length", settings.default_sequence_length))
    )

    df = data_store.get()
    if len(df) <= sequence_length:
        raise HTTPException(
            status_code=400,
            detail=f"데이터 회차 수({len(df)})가 sequence_length({sequence_length})보다 작거나 같습니다.",
        )

    try:
        result = make_prediction(
            model=model,
            scaler=scaler,
            df=df,
            sequence_length=sequence_length,
            seed=body.seed,
            num_sets=body.num_sets,
        )
    except Exception as e:
        logger.exception("예측 실패")
        raise HTTPException(status_code=500, detail=f"예측 실패: {e}")

    return PredictResponse(
        lstm=result.lstm,
        ensemble=result.ensemble,
        additional_sets=result.additional_sets,
        active_bundle=active_name,
        data_hash_match=data_hash_matches(meta, data_store.data_path),
        sequence_length=result.sequence_length,
        seed=result.seed,
    )
