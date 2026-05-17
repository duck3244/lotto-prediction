"""
백엔드 절대 경로 정의 모듈.

이 모듈은 모든 산출물 디렉토리와 기본 입력 파일을 ``backend/`` 기준 절대경로로
노출한다. CLI/uvicorn 어느 워킹 디렉토리에서 실행되어도 동일한 위치에 파일이
생성/조회되어야 하므로, ``Path("models")``처럼 상대경로를 직접 만들지 말고
반드시 본 모듈의 상수를 사용한다.

부수효과(디렉토리 생성 등)는 두지 않는다. 호출 측에서 ``mkdir(exist_ok=True)``
하여 사용한다.
"""

from __future__ import annotations

from pathlib import Path

# backend/ 자체를 기준 경로로 한다.
BASE_DIR: Path = Path(__file__).resolve().parent

# 런타임 산출물 디렉토리
LOGS_DIR: Path = BASE_DIR / "logs"
MODELS_DIR: Path = BASE_DIR / "models"
VISUALIZATION_DIR: Path = BASE_DIR / "visualization"
TENSORBOARD_DIR: Path = LOGS_DIR / "tensorboard"

# 기본 데이터/리포트 경로
DEFAULT_DATA_FILE: Path = BASE_DIR / "lotto.xlsx"
EVALUATION_PLOT_PATH: Path = VISUALIZATION_DIR / "model_evaluation.png"
LEARNING_CURVE_PATH: Path = VISUALIZATION_DIR / "learning_curve.png"

# 기본 quantized 모델 저장 경로 (확장자는 호출 측에서 부여)
DEFAULT_QUANTIZED_MODEL_BASENAME: Path = MODELS_DIR / "quantized_model"
