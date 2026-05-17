"""FastAPI 라우터에서 사용하는 공용 의존성.

``DataStore`` / ``ModelStore`` 는 lifespan 에서 단일 인스턴스를 만들어
``app.state`` 에 보관한다. 라우터는 ``Depends(get_data_store)`` /
``Depends(get_model_store)`` 로 꺼내 쓴다.
"""

from __future__ import annotations

from fastapi import Request

from app.services.data_store import DataStore
from app.services.model_store import ModelStore


def get_data_store(request: Request) -> DataStore:
    return request.app.state.data_store


def get_model_store(request: Request) -> ModelStore:
    return request.app.state.model_store
