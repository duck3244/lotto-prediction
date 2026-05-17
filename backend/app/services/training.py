"""학습 잡 실행 서비스.

단일 사용자 MVP: ``ThreadPoolExecutor(max_workers=1)`` 으로 동시 학습을 1건으로
직렬화한다. 진행 상태는 ``threading.RLock`` 으로 보호된 메모리 딕셔너리에 보관한다.
**프로세스 재기동 시 휘발됨** — 영속화가 필요해지면 SQLite/파일 기반으로 바꾼다.

설계 메모:
- 학습 중에는 GPU 메모리가 거의 모두 점유되므로 동일 프로세스의 ``/api/predict``
  는 ``409 Conflict`` 로 거절한다(라우터에서 ``is_running()`` 확인).
- ``auto_activate=True`` 면 완료 후 새 번들을 활성 모델로 즉시 교체한다.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

import tensorflow as tf

from data_loader import preprocess_data
from model import train_and_evaluate
from paths import DEFAULT_DATA_FILE
from utils import save_training_bundle, set_global_seeds

from app.services.data_store import DataStore
from app.services.model_store import ModelStore

logger = logging.getLogger(__name__)


@dataclass
class TrainingParams:
    epochs: int = 300
    batch_size: int = 64
    sequence_length: int = 10
    seed: int = 42
    auto_activate: bool = True


@dataclass
class TrainingJob:
    job_id: str
    params: TrainingParams
    status: str = "queued"  # queued | running | completed | failed
    epoch: int = 0
    total_epochs: int = 0
    best_val_loss: Optional[float] = None
    last_loss: Optional[float] = None
    last_val_loss: Optional[float] = None
    error: Optional[str] = None
    bundle_name: Optional[str] = None
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "best_val_loss": self.best_val_loss,
            "last_loss": self.last_loss,
            "last_val_loss": self.last_val_loss,
            "error": self.error,
            "bundle_name": self.bundle_name,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "params": {
                "epochs": self.params.epochs,
                "batch_size": self.params.batch_size,
                "sequence_length": self.params.sequence_length,
                "seed": self.params.seed,
                "auto_activate": self.params.auto_activate,
            },
        }


class _EpochProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, job: TrainingJob) -> None:
        super().__init__()
        self._job = job

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:  # type: ignore[override]
        logs = logs or {}
        self._job.epoch = int(epoch) + 1
        if "loss" in logs:
            self._job.last_loss = float(logs["loss"])
        if "val_loss" in logs:
            vl = float(logs["val_loss"])
            self._job.last_val_loss = vl
            if self._job.best_val_loss is None or vl < self._job.best_val_loss:
                self._job.best_val_loss = vl


class TrainingService:
    def __init__(self, data_store: DataStore, model_store: ModelStore) -> None:
        self._data_store = data_store
        self._model_store = model_store
        self._jobs: dict[str, TrainingJob] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="training")
        self._current_job_id: Optional[str] = None

    # ---- 조회 ----

    def get(self, job_id: str) -> Optional[TrainingJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[TrainingJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.submitted_at, reverse=True)

    def is_running(self) -> bool:
        with self._lock:
            jid = self._current_job_id
            if not jid:
                return False
            job = self._jobs.get(jid)
            return bool(job and job.status == "running")

    def current_job(self) -> Optional[TrainingJob]:
        with self._lock:
            return self._jobs.get(self._current_job_id) if self._current_job_id else None

    # ---- 제출/실행 ----

    def submit(self, params: TrainingParams) -> TrainingJob:
        job_id = uuid.uuid4().hex[:12]
        job = TrainingJob(job_id=job_id, params=params, total_epochs=params.epochs)
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(self._run, job)
        logger.info("학습 잡 제출: %s (epochs=%d, batch=%d, seq=%d)",
                    job_id, params.epochs, params.batch_size, params.sequence_length)
        return job

    def _run(self, job: TrainingJob) -> None:
        with self._lock:
            self._current_job_id = job.job_id
            job.status = "running"
            job.started_at = time.time()
        try:
            # 결정성/재현성: shuffle / dropout / glorot init 등 TF random ops 가
            # 호출되기 전에 global seed 를 세팅한다 (TF_DETERMINISTIC_OPS=1 필수).
            set_global_seeds(job.params.seed)

            df = self._data_store.get()
            if len(df) <= job.params.sequence_length:
                raise ValueError(
                    f"데이터 회차({len(df)})가 sequence_length({job.params.sequence_length})보다 작거나 같음"
                )

            X, y, scaler = preprocess_data(df, job.params.sequence_length, use_tf_dataset=True)

            cb = _EpochProgressCallback(job)
            model, _ = train_and_evaluate(
                X,
                y,
                epochs=job.params.epochs,
                batch_size=job.params.batch_size,
                extra_callbacks=[cb],
            )

            bundle_dir = save_training_bundle(
                model,
                scaler,
                data_path=DEFAULT_DATA_FILE,
                sequence_length=job.params.sequence_length,
                seed=job.params.seed,
                extra_meta={
                    "epochs": job.params.epochs,
                    "batch_size": job.params.batch_size,
                    "best_val_loss": job.best_val_loss,
                    "trained_via": "api",
                },
            )

            with self._lock:
                job.bundle_name = bundle_dir.name
                job.status = "completed"
                job.finished_at = time.time()

            if job.params.auto_activate:
                try:
                    self._model_store.activate(bundle_dir.name)
                    logger.info("학습 잡 %s 완료 → 활성 모델 자동 교체: %s", job.job_id, bundle_dir.name)
                except Exception:
                    logger.exception("자동 활성화 실패: %s", bundle_dir.name)

        except Exception as e:  # 학습 자체 오류 / 데이터 오류
            logger.exception("학습 잡 %s 실패", job.job_id)
            with self._lock:
                job.status = "failed"
                job.error = str(e)
                job.finished_at = time.time()
        finally:
            with self._lock:
                self._current_job_id = None
