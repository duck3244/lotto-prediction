"""학습 잡 라우터 — `/api/train`."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from app.schemas import TrainJob, TrainJobListResponse, TrainRequest
from app.services.training import TrainingParams, TrainingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/train", tags=["train"])


def get_training_service(request: Request) -> TrainingService:
    return request.app.state.training_service


@router.post("", response_model=TrainJob)
def submit(
    body: TrainRequest,
    service: TrainingService = Depends(get_training_service),
) -> TrainJob:
    if service.is_running():
        raise HTTPException(status_code=409, detail="다른 학습 잡이 진행 중입니다.")
    params = TrainingParams(
        epochs=body.epochs,
        batch_size=body.batch_size,
        sequence_length=body.sequence_length,
        seed=body.seed,
        auto_activate=body.auto_activate,
    )
    job = service.submit(params)
    return TrainJob(**job.to_dict())


@router.get("", response_model=TrainJobListResponse)
def list_jobs(service: TrainingService = Depends(get_training_service)) -> TrainJobListResponse:
    current = service.current_job()
    return TrainJobListResponse(
        current_job_id=current.job_id if current else None,
        jobs=[TrainJob(**j.to_dict()) for j in service.list()],
    )


@router.get("/{job_id}", response_model=TrainJob)
def get_job(
    job_id: str,
    service: TrainingService = Depends(get_training_service),
) -> TrainJob:
    job = service.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"존재하지 않는 잡: {job_id}")
    return TrainJob(**job.to_dict())
