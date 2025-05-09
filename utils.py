"""
유틸리티 및 헬퍼 함수 모듈 (RTX 4060 최적화)
- 모델 저장 및 로드
- GPU 모니터링
- 결과 검증
"""

import os
from pathlib import Path
import subprocess
import time
import threading
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model

def save_model(model, file_path):
    """
    학습된 모델을 파일로 저장

    Args:
        model: 저장할 Keras 모델
        file_path: 저장 경로

    Returns:
        성공 여부
    """
    try:
        # 저장 경로의 디렉토리 생성
        save_dir = os.path.dirname(file_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        model.save(file_path)
        return True
    except Exception as e:
        print(f"모델 저장 오류: {e}")
        return False

def load_model(file_path):
    """
    저장된 모델 불러오기

    Args:
        file_path: 모델 파일 경로

    Returns:
        불러온 모델 또는 None
    """
    try:
        if os.path.exists(file_path):
            model = keras_load_model(file_path)
            return model
        else:
            print(f"모델 파일 '{file_path}'을 찾을 수 없습니다.")
            return None
    except Exception as e:
        print(f"모델 로드 오류: {e}")
        return None


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
    # 6개의 번호인지 확인
    if len(numbers) != 6:
        return False, "로또 번호는 정확히 6개여야 합니다."

    # 번호가 1-45 사이인지 확인
    for num in numbers:
        if not (1 <= num <= 45):
            return False, f"번호 {num}은(는) 유효하지 않습니다. 모든 번호는 1-45 사이여야 합니다."

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

    for _ in range(num_to_generate):
        # 1. 빈도 기반 (40%)
        high_freq = [num for num, _ in sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:15]]

        # 2. 저빈도 번호 (20%)
        low_freq = [num for num, _ in sorted(frequencies.items(), key=lambda x: x[1])[:15]]

        # 3. 최근에 나온 번호 (20%)
        recent = list(recent_numbers)[:15]

        # 4. 랜덤 번호 (20%)
        random_nums = np.random.choice([n for n in range(1, 46) if n not in high_freq[:6]], 10, replace=False)

        # 비율에 맞게 선택
        np.random.shuffle(high_freq)
        np.random.shuffle(low_freq)
        np.random.shuffle(recent)

        selected = []
        # 빈도 높은 번호에서 3개
        selected.extend(high_freq[:3])
        # 빈도 낮은 번호에서 1개
        selected.extend(low_freq[:1])
        # 최근 번호에서 1개
        if recent:
            selected.extend(recent[:1])
        # 랜덤 번호에서 나머지
        needed = 6 - len(selected)
        selected.extend(random_nums[:needed])

        # 정렬 및 중복 제거
        balanced_set = sorted(list(set(selected)))

        # 정확히 6개 번호 맞추기
        while len(balanced_set) < 6:
            missing = 6 - len(balanced_set)
            available = [n for n in range(1, 46) if n not in balanced_set]
            balanced_set.extend(np.random.choice(available, missing, replace=False))
            balanced_set = sorted(list(set(balanced_set)))

        # 6개를 초과할 경우 자르기
        if len(balanced_set) > 6:
            balanced_set = sorted(balanced_set[:6])

        balanced_sets.append(balanced_set)

    return balanced_sets


def optimize_tf_config():
    """TensorFlow 설정 최적화 (RTX 4060 GPU 대상)"""
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
                # RTX 40 시리즈 최적화
                tf.config.threading.set_inter_op_parallelism_threads(4)
                tf.config.threading.set_intra_op_parallelism_threads(8)

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