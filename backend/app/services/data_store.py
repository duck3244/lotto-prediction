"""로또 데이터 캐시.

``lotto.xlsx`` 를 한 번 로드해 메모리에 유지하고, 명시적 ``reload()`` 호출 시에만
다시 읽는다. V3 업로드 기능이 들어오면 업로드 핸들러가 ``reload()`` 를 호출해
갱신한다.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import pandas as pd

from data_loader import load_data
from paths import DEFAULT_DATA_FILE

logger = logging.getLogger(__name__)


class DataStore:
    def __init__(self, data_path: Path = DEFAULT_DATA_FILE) -> None:
        self._data_path = data_path
        self._df: Optional[pd.DataFrame] = None
        self._lock = threading.Lock()

    @property
    def data_path(self) -> Path:
        return self._data_path

    def get(self) -> pd.DataFrame:
        with self._lock:
            if self._df is None:
                logger.info("로또 데이터 로드: %s", self._data_path)
                self._df = load_data(str(self._data_path))
            return self._df

    def reload(self) -> pd.DataFrame:
        with self._lock:
            logger.info("로또 데이터 재로드: %s", self._data_path)
            self._df = load_data(str(self._data_path))
            return self._df
