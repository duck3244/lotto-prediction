"""
유틸리티 및 헬퍼 함수 모듈 (RTX 4060 최적화)
- 모델 저장 및 로드
- GPU 모니터링
- 결과 검증
"""

import gc
import hashlib
import json
import os
import time
import pickle
import subprocess
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import sklearn
import tensorflow as tf

from tensorflow.keras.models import load_model as keras_load_model

from constants import NUM_BALLS, MIN_NUMBER, MAX_NUMBER
from paths import MODELS_DIR


def set_global_seeds(seed=42):
    """
    런타임 랜덤 시드 설정. ``random`` / ``numpy`` / ``tf`` 의 시드만 다룬다.

    결정성에 영향을 주는 환경변수(``TF_DETERMINISTIC_OPS`` 등)는
    ``env_setup.set_deterministic_env()`` 가 ``import tensorflow`` 이전에
    설정해두어야 한다 — 본 함수는 그 환경변수를 다루지 않는다.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _tf_supports_keras_format() -> bool:
    """현재 TensorFlow가 네이티브 ``.keras`` 포맷을 지원하는지 확인 (TF 2.13+)."""
    try:
        major, minor, *_ = (int(p) for p in tf.__version__.split(".")[:2])
        return (major, minor) >= (2, 13)
    except Exception:
        return False


def preferred_model_extension() -> str:
    """TF가 지원하면 ``.keras``, 아니면 레거시 ``.h5``를 반환."""
    return ".keras" if _tf_supports_keras_format() else ".h5"


def save_model(model, file_path: str) -> bool:
    """
    학습된 모델을 파일로 저장. TF가 ``.keras``를 지원하지 않으면 ``.h5``로 폴백한다.
    """
    try:
        save_dir = os.path.dirname(file_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # .keras 확장자인데 현재 TF에서 지원하지 않으면 .h5로 폴백
        if file_path.endswith(".keras") and not _tf_supports_keras_format():
            file_path = file_path[: -len(".keras")] + ".h5"
            print(f"현재 TensorFlow는 .keras 포맷을 지원하지 않습니다. {file_path}로 저장합니다.")

        model.save(file_path)
        return True
    except Exception as e:
        print(f"모델 저장 오류: {e}")
        return False


def save_scaler(scaler, file_path):
    """
    학습에 사용된 스케일러를 파일로 저장

    Args:
        scaler: 저장할 MinMaxScaler 객체
        file_path: 저장 경로 (.pkl)

    Returns:
        성공 여부
    """
    try:
        save_dir = os.path.dirname(file_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(scaler, f)
        return True
    except Exception as e:
        print(f"스케일러 저장 오류: {e}")
        return False


def load_scaler(file_path):
    """
    저장된 스케일러 불러오기

    Args:
        file_path: 스케일러 파일 경로 (.pkl)

    Returns:
        불러온 스케일러 또는 None
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"스케일러 파일 '{file_path}'을 찾을 수 없습니다.")
            return None
    except Exception as e:
        print(f"스케일러 로드 오류: {e}")
        return None


def load_model(file_path):
    """
    저장된 모델 불러오기

    Args:
        file_path: 모델 파일 경로

    Returns:
        불러온 모델 또는 None
    """
    try:
        if not os.path.exists(file_path):
            print(f"모델 파일 '{file_path}'을 찾을 수 없습니다.")
            return None

        if file_path.endswith('.h5'):
            print("레거시 .h5 형식 모델을 로드합니다. .keras 형식으로의 재저장을 권장합니다.")

        model = keras_load_model(file_path)
        return model
    except Exception as e:
        print(f"모델 로드 오류: {e}")
        return None


def release_model_resources() -> None:
    """현재 프로세스에 적재된 Keras/TensorFlow 세션을 해제하고 GC를 강제한다.

    FastAPI에서 활성 모델을 교체하기 직전에 호출한다. ``clear_session()`` 만으로는
    파이썬 객체(``Model`` 인스턴스)가 살아 있으면 GPU 메모리가 즉시 반환되지 않으므로,
    호출 측은 모델/스케일러 참조를 ``None`` 으로 떨군 뒤 본 함수를 호출하는 패턴을
    권장한다.

    예::

        self._active_model = None
        self._active_scaler = None
        release_model_resources()
    """
    try:
        tf.keras.backend.clear_session()
    except Exception as e:  # pragma: no cover - 방어적
        print(f"clear_session 경고: {e}")
    gc.collect()


# --- 학습 산출물 번들 (모델 + 스케일러 + 메타데이터) ---------------------------

_BUNDLE_META_FILENAME = "meta.json"
_BUNDLE_SCALER_FILENAME = "scaler.pkl"


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """파일 내용의 SHA-256 16진 다이제스트."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def save_training_bundle(
    model,
    scaler,
    *,
    data_path: Path | str,
    sequence_length: int,
    seed: int,
    bundle_dir: Optional[Path] = None,
    extra_meta: Optional[dict[str, Any]] = None,
) -> Path:
    """학습 산출물을 하나의 디렉토리로 묶어 저장한다.

    ``models/bundle_<YYYYmmdd_HHMMSS>/`` 아래에 다음을 저장한다::

        bundle_<ts>/
            model.keras      # (TF<2.13 이면 model.h5)
            scaler.pkl
            meta.json        # 데이터 해시, sequence_length, seed, 버전 등

    Returns:
        생성된 번들 디렉토리 절대경로.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if bundle_dir is None:
        bundle_dir = MODELS_DIR / f"bundle_{timestamp}"
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    model_filename = f"model{preferred_model_extension()}"
    model_path = bundle_dir / model_filename
    if not save_model(model, str(model_path)):
        raise RuntimeError(f"모델 저장 실패: {model_path}")
    # save_model() 이 .keras→.h5 폴백을 했을 수 있으므로 실제 저장된 파일명을 다시 탐색
    actual_model = next(
        (p for p in (bundle_dir / "model.keras", bundle_dir / "model.h5") if p.exists()),
        model_path,
    )

    scaler_path = bundle_dir / _BUNDLE_SCALER_FILENAME
    if not save_scaler(scaler, str(scaler_path)):
        raise RuntimeError(f"스케일러 저장 실패: {scaler_path}")

    data_path = Path(data_path)
    meta: dict[str, Any] = {
        "schema_version": 1,
        "timestamp": timestamp,
        "model_file": actual_model.name,
        "scaler_file": scaler_path.name,
        "data_file": str(data_path),
        "data_sha256": _sha256_file(data_path) if data_path.exists() else None,
        "sequence_length": int(sequence_length),
        "seed": int(seed),
        "tensorflow_version": tf.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }
    if extra_meta:
        meta.update(extra_meta)

    (bundle_dir / _BUNDLE_META_FILENAME).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return bundle_dir


def load_training_bundle(bundle_dir: Path | str):
    """:func:`save_training_bundle` 로 저장된 번들을 읽어 ``(model, scaler, meta)`` 반환.

    번들 무결성 점검:
    - ``meta.json`` 의 ``model_file`` / ``scaler_file`` 이 디렉토리에 존재해야 한다.
    - 모델/스케일러 로드에 실패하면 ``FileNotFoundError`` 를 던진다.

    데이터 해시 검증은 호출 측의 책임 — 현재 ``data_file`` 이 ``meta["data_sha256"]`` 과
    다르면 경고 후 진행할지, 거부할지 정책이 갈리기 때문에 본 함수는 정보만 반환한다.
    """
    bundle_dir = Path(bundle_dir)
    meta_path = bundle_dir / _BUNDLE_META_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(f"번들 메타데이터를 찾을 수 없음: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    model_path = bundle_dir / meta["model_file"]
    scaler_path = bundle_dir / meta["scaler_file"]
    if not model_path.exists():
        raise FileNotFoundError(f"번들 모델 파일 없음: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"번들 스케일러 파일 없음: {scaler_path}")

    model = load_model(str(model_path))
    if model is None:
        raise RuntimeError(f"모델 로드 실패: {model_path}")
    scaler = load_scaler(str(scaler_path))
    if scaler is None:
        raise RuntimeError(f"스케일러 로드 실패: {scaler_path}")

    return model, scaler, meta


def data_hash_matches(meta: dict[str, Any], data_path: Path | str) -> bool:
    """현재 데이터 파일이 번들 메타에 기록된 SHA-256 과 동일한지 검사.

    메타에 ``data_sha256`` 이 없거나 파일이 없으면 ``False`` 를 반환한다.
    호출 측은 이 결과를 UI 배지/경고로 활용할 수 있다.
    """
    expected = meta.get("data_sha256")
    if not expected:
        return False
    data_path = Path(data_path)
    if not data_path.exists():
        return False
    return _sha256_file(data_path) == expected


def list_training_bundles(models_root: Optional[Path] = None) -> list[Path]:
    """``models/`` 아래의 학습 번들 디렉토리를 신규→과거 순으로 정렬해 반환.

    번들로 간주되는 조건: ``meta.json`` 이 존재.
    """
    root = Path(models_root) if models_root is not None else MODELS_DIR
    if not root.exists():
        return []
    bundles = [p for p in root.iterdir() if p.is_dir() and (p / _BUNDLE_META_FILENAME).exists()]
    bundles.sort(key=lambda p: p.name, reverse=True)
    return bundles


def setup_gpu_monitoring(interval=5):
    """
    RTX 4060 GPU 사용량 모니터링 설정

    Args:
        interval: 모니터링 간격 (초)

    Returns:
        monitor_thread: 모니터링 스레드
        stop_monitoring: 모니터링 중지 함수
    """
    monitoring = {'active': True}  # 딕셔너리로 상태 관리 (스레드 간 공유)


    def gpu_monitor():
        try:
            while monitoring['active']:
                try:
                    # nvidia-smi 명령으로 GPU 정보 수집
                    result = subprocess.run(
                        ['nvidia-smi',
                         '--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total',
                         '--format=csv,noheader'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True
                    )

                    print("\n--- GPU 모니터링 정보 ---")
                    # 출력 정리
                    gpu_info = result.stdout.strip().split(',')
                    if len(gpu_info) >= 7:
                        print(f"GPU: {gpu_info[1].strip()}")
                        print(f"온도: {gpu_info[2].strip()}°C")
                        print(f"GPU 사용률: {gpu_info[3].strip()}")
                        print(f"메모리 사용률: {gpu_info[4].strip()}")
                        print(f"메모리: {gpu_info[5].strip()} / {gpu_info[6].strip()}")
                    else:
                        print(result.stdout)
                    print("------------------------\n")

                except subprocess.SubprocessError:
                    print("GPU 정보를 가져올 수 없습니다.")

                time.sleep(interval)
        except Exception as e:
            print(f"GPU 모니터링 오류: {e}")

    # 모니터링 스레드 시작
    monitor_thread = threading.Thread(target=gpu_monitor)
    monitor_thread.daemon = True
    monitor_thread.start()


    # 모니터링 중지 함수
    def stop_monitoring():
        monitoring['active'] = False
        monitor_thread.join(timeout=1)
        print("GPU 모니터링이 중지되었습니다.")

    return monitor_thread, stop_monitoring


def compare_predictions(actual_numbers, predicted_numbers):
    """
    예측 번호와 실제 당첨 번호 비교

    Args:
        actual_numbers: 실제 당첨 번호 리스트
        predicted_numbers: 예측 번호 리스트

    Returns:
        matches_count: 일치하는 번호 개수
        matched_numbers: 일치하는 번호 리스트
    """
    if not actual_numbers or not predicted_numbers:
        return 0, []

    # 일치하는 번호 찾기
    matches = set(actual_numbers) & set(predicted_numbers)

    return len(matches), list(sorted(matches))


def validate_lotto_numbers(numbers):
    """
    로또 번호 유효성 검사

    Args:
        numbers: 확인할 로또 번호 리스트

    Returns:
        is_valid: 유효성 여부
        message: 결과 메시지
    """
    # 번호 개수 확인
    if len(numbers) != NUM_BALLS:
        return False, f"로또 번호는 정확히 {NUM_BALLS}개여야 합니다."

    # 번호 범위 확인
    for num in numbers:
        if not (MIN_NUMBER <= num <= MAX_NUMBER):
            return False, f"번호 {num}은(는) 유효하지 않습니다. 모든 번호는 {MIN_NUMBER}-{MAX_NUMBER} 사이여야 합니다."

    # 중복 번호 확인
    if len(set(numbers)) != len(numbers):
        return False, "중복된 번호가 있습니다."

    return True, "유효한 로또 번호입니다."


def suggest_balanced_numbers(frequencies, recent_numbers, num_to_generate=5):
    """
    통계적으로 균형 잡힌 번호 조합 제안

    Args:
        frequencies: 번호별 출현 빈도
        recent_numbers: 최근에 나온 번호들의 집합
        num_to_generate: 생성할 번호 조합 개수

    Returns:
        balanced_sets: 생성된 번호 조합 리스트
    """
    balanced_sets = []
    sorted_high = [num for num, _ in sorted(frequencies.items(), key=lambda x: x[1], reverse=True)]
    sorted_low = [num for num, _ in sorted(frequencies.items(), key=lambda x: x[1])]

    for _ in range(num_to_generate):
        high_freq = sorted_high[:15].copy()
        low_freq = sorted_low[:15].copy()
        recent = list(recent_numbers)[:15]

        np.random.shuffle(high_freq)
        np.random.shuffle(low_freq)
        np.random.shuffle(recent)

        # 빈도 상위 3개 + 저빈도 1개 + 최근 1개 + 랜덤으로 나머지
        selected: list[int] = []
        selected.extend(high_freq[:3])
        for n in low_freq:
            if n not in selected:
                selected.append(n)
                break
        for n in recent:
            if n not in selected:
                selected.append(n)
                break

        while len(set(selected)) < NUM_BALLS:
            available = [n for n in range(MIN_NUMBER, MAX_NUMBER + 1) if n not in selected]
            selected.append(int(np.random.choice(available)))

        balanced_sets.append(sorted(set(selected)))

    return balanced_sets


def setup_gpu():
    """RTX 4060에 최적화된 GPU 설정 (메모리 증가, 혼합 정밀도, XLA 컴파일)"""
    try:
        # GPU 메모리 증가 설정
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # 혼합 정밀도 계산 활성화
            tf.keras.mixed_precision.set_global_policy('mixed_float16')

            # XLA 컴파일 활성화
            tf.config.optimizer.set_jit(True)

            # GPU 유형에 따른 추가 최적화
            device_details = tf.config.experimental.get_device_details(gpus[0])
            device_name = device_details.get('device_name', '').lower()

            if 'rtx' in device_name and (
                    '4060' in device_name or '4070' in device_name or '4080' in device_name or '4090' in device_name):
                # RTX 40 시리즈 최적화: CPU 코어 수 기반 동적 스레드 설정
                cpu_cnt = os.cpu_count() or 8
                tf.config.threading.set_inter_op_parallelism_threads(max(2, cpu_cnt // 4))
                tf.config.threading.set_intra_op_parallelism_threads(max(4, cpu_cnt // 2))

            return True, f"TensorFlow GPU 설정 최적화 완료: {device_name}"
    except Exception as e:
        return False, f"TensorFlow 설정 오류: {e}"

    return False, "GPU를 찾을 수 없습니다. CPU 모드로 실행됩니다."


def calculate_win_probability(predictions, actual_results):
    """
    예측 번호의 실제 당첨 확률 계산

    Args:
        predictions: 예측 번호 리스트의 리스트 (여러 회차의 예측)
        actual_results: 실제 당첨 번호 리스트의 리스트 (여러 회차의 결과)

    Returns:
        match_stats: 일치 개수별 통계
        win_probability: 3개 이상 일치 확률
    """
    match_stats = {i: 0 for i in range(7)}
    total = min(len(predictions), len(actual_results))

    if total == 0:
        return match_stats, 0

    for pred, actual in zip(predictions, actual_results):
        match_count, _ = compare_predictions(pred, actual)
        match_stats[match_count] += 1

    # 3개 이상 일치 확률
    win_count = sum(match_stats[i] for i in range(3, 7))
    win_probability = win_count / total

    return match_stats, win_probability

