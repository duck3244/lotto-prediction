"""
TensorFlow 결정성/스레드 관련 환경변수 설정 모듈.

**이 모듈은 TensorFlow를 import하지 않는다.** ``set_deterministic_env()``는
``import tensorflow`` 이전에 호출되어야 효력이 있으므로, 모든 진입점
(``main.py``, ``predict.py``, ``evaluate.py``, 향후 ``app/main.py``)에서
**가장 먼저** 호출되어야 한다.

``PYTHONHASHSEED``는 인터프리터 시작 시점에 읽히므로 실행 중 ``os.environ``으로
세팅해도 현재 프로세스에는 효과가 없다(파생 프로세스에만 전달됨). 따라서 본
모듈은 PYTHONHASHSEED를 강제하지 않는다. 완전한 결정성이 필요하면 셸에서
``PYTHONHASHSEED=0 python ...`` 형태로 실행한다.
"""

from __future__ import annotations

import os


def set_deterministic_env() -> None:
    """TF 결정성에 영향을 주는 환경변수를 설정한다.

    반드시 ``import tensorflow`` 이전에 호출되어야 한다. 이미 다른 곳에서
    값을 설정해두었다면 덮어쓰지 않는다(``setdefault``).
    """
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
