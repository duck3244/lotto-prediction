"""학습 번들 / 활성 모델 라우터 — `/api/models/*`."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.deps import get_data_store, get_model_store
from app.schemas import (
    ActivateBundleRequest,
    ActivateBundleResponse,
    BundleSummary,
    ModelsResponse,
)
from app.services.data_store import DataStore
from app.services.model_store import ModelStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelsResponse)
def list_models(
    data_store: DataStore = Depends(get_data_store),
    model_store: ModelStore = Depends(get_model_store),
) -> ModelsResponse:
    items = model_store.list_bundles(data_store.data_path)
    bundles = [
        BundleSummary(
            name=item["name"],
            timestamp=item["meta"].get("timestamp"),
            sequence_length=item["meta"].get("sequence_length"),
            seed=item["meta"].get("seed"),
            data_sha256=item["meta"].get("data_sha256"),
            tensorflow_version=item["meta"].get("tensorflow_version"),
            sklearn_version=item["meta"].get("sklearn_version"),
            is_active=item["is_active"],
            data_hash_match=item["data_hash_match"],
        )
        for item in items
    ]
    return ModelsResponse(bundles=bundles, active_name=model_store.active_name)


@router.post("/active", response_model=ActivateBundleResponse)
def activate(
    body: ActivateBundleRequest,
    data_store: DataStore = Depends(get_data_store),
    model_store: ModelStore = Depends(get_model_store),
) -> ActivateBundleResponse:
    try:
        meta, _, _ = model_store.activate(body.name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # 로드/세션 정리 실패
        logger.exception("번들 활성화 실패: %s", body.name)
        raise HTTPException(status_code=500, detail=f"활성화 실패: {e}")

    hash_match = True
    try:
        from utils import data_hash_matches

        hash_match = data_hash_matches(meta, data_store.data_path)
    except Exception:
        hash_match = False

    message = (
        "활성 번들 변경 완료."
        if hash_match
        else "활성 번들 변경 완료 — 단, 학습 시점의 데이터 SHA-256 과 현재 lotto.xlsx 의 해시가 다릅니다."
    )

    return ActivateBundleResponse(
        active_name=body.name,
        data_hash_match=hash_match,
        message=message,
    )
