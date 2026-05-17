"""활성 모델/스케일러/메타 상태 관리.

단일 사용자 MVP라 동시 요청은 적지만, FastAPI threadpool 에서 sync route 가 병렬로
실행될 수 있어 자원 보호용 ``threading.RLock`` 을 둔다. 모델 교체 시
- 기존 모델/스케일러 참조를 ``None`` 으로 떨구고
- ``release_model_resources()`` 로 ``clear_session()`` + ``gc.collect()`` 호출
- 새 번들 로드
순서로 동작해 GPU 메모리 누수를 막는다.

활성 번들은 ``models/active.json`` 에 ``{"bundle_name": "bundle_..."}`` 로 영속화된다.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional

from paths import MODELS_DIR
from utils import (
    data_hash_matches,
    list_training_bundles,
    load_training_bundle,
    release_model_resources,
)

logger = logging.getLogger(__name__)

_ACTIVE_STATE_FILE = MODELS_DIR / "active.json"


class ModelStore:
    """활성 모델 상태를 보관/교체하는 서비스."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model = None
        self._scaler = None
        self._meta: Optional[dict[str, Any]] = None
        self._active_name: Optional[str] = None

    # ---- 조회 ----

    @property
    def active_name(self) -> Optional[str]:
        with self._lock:
            return self._active_name

    def snapshot(self) -> tuple[Any, Any, Optional[dict[str, Any]], Optional[str]]:
        """현재 활성 (model, scaler, meta, name) 의 스냅샷.

        이 함수가 반환한 모델 객체는 호출 측에서 사용 중일 동안 ModelStore 가
        다른 모델로 교체될 수 있으므로, 예측 한 사이클 내에서만 사용한다.
        """
        with self._lock:
            return self._model, self._scaler, self._meta, self._active_name

    def list_bundles(self, data_path: Path) -> list[dict[str, Any]]:
        """``models/`` 아래의 번들 목록 + 메타 + 활성/해시매치 플래그."""
        bundles_meta: list[dict[str, Any]] = []
        with self._lock:
            active = self._active_name
        for bundle_dir in list_training_bundles():
            meta_file = bundle_dir / "meta.json"
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("번들 메타 파싱 실패 %s: %s", meta_file, e)
                continue
            bundles_meta.append(
                {
                    "name": bundle_dir.name,
                    "meta": meta,
                    "is_active": bundle_dir.name == active,
                    "data_hash_match": data_hash_matches(meta, data_path),
                }
            )
        return bundles_meta

    # ---- 교체 ----

    def activate(self, name: str) -> tuple[dict[str, Any], bool, Path]:
        """이름(번들 디렉토리)으로 활성 모델을 교체한다.

        경로 traversal 방지를 위해 ``MODELS_DIR`` 내부의 직계 자식 디렉토리만 허용.
        Returns ``(meta, data_hash_match, bundle_dir)``.
        """
        bundle_dir = (MODELS_DIR / name).resolve()
        models_root = MODELS_DIR.resolve()
        if not bundle_dir.is_dir() or bundle_dir.parent != models_root:
            raise FileNotFoundError(f"잘못된 번들 이름: {name}")

        with self._lock:
            # 이전 모델 자원 해제
            self._model = None
            self._scaler = None
            self._meta = None
            self._active_name = None
            release_model_resources()

            model, scaler, meta = load_training_bundle(bundle_dir)
            self._model = model
            self._scaler = scaler
            self._meta = meta
            self._active_name = bundle_dir.name

            self._persist()
            logger.info("활성 번들 변경: %s", bundle_dir.name)
            return meta, data_hash_matches(meta, _data_path_for_hash_check()), bundle_dir

    def restore_from_disk(self) -> Optional[str]:
        """``active.json`` 의 마지막 활성 번들을 다시 로드한다 (lifespan startup용).

        실패하면 활성 없음 상태로 둔다. 반환값은 활성화된 번들 이름(또는 None).
        """
        if not _ACTIVE_STATE_FILE.exists():
            return None
        try:
            state = json.loads(_ACTIVE_STATE_FILE.read_text(encoding="utf-8"))
            name = state.get("bundle_name")
            if not name:
                return None
            self.activate(name)
            return self.active_name
        except Exception as e:
            logger.warning("이전 활성 번들 복원 실패: %s", e)
            return None

    # ---- 내부 ----

    def _persist(self) -> None:
        MODELS_DIR.mkdir(exist_ok=True)
        _ACTIVE_STATE_FILE.write_text(
            json.dumps({"bundle_name": self._active_name}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _data_path_for_hash_check() -> Path:
    """activate 호출 시점에 비교할 데이터 파일 경로."""
    from paths import DEFAULT_DATA_FILE

    return DEFAULT_DATA_FILE
