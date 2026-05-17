"""pytest 공통 설정.

- ``backend/`` 디렉토리를 ``sys.path`` 에 추가해 flat 임포트(``from utils import …``)가
  pytest 컬렉션 단계에서도 정상 동작하도록 한다.
- TF 결정성 환경변수를 가장 먼저 세팅한다 (``utils.py`` 가 ``tensorflow`` 를 import
  하므로 conftest 가 처음 로드될 때 env 가 잡혀 있어야 한다).
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# 결정성 환경변수 — TF import 이전에 세팅되어야 한다.
from env_setup import set_deterministic_env  # noqa: E402

set_deterministic_env()
