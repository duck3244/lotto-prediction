"""
데이터 로딩 및 전처리 모듈 (RTX 4060 최적화)
- 엑셀 파일에서 로또 데이터 로드
- LSTM 모델용 시퀀스 데이터 생성
- TensorFlow 데이터셋 최적화
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """엑셀 파일에서 로또 데이터 불러오기"""
    try:
        df = pd.read_excel(file_path)
        # 회차 기준으로 내림차순 정렬 (최근 데이터가 먼저 오도록)
        df = df.sort_values('회차', ascending=False).reset_index(drop=True)
        return df
    except Exception as e:
        raise FileNotFoundError(f"데이터 파일을 불러올 수 없습니다: {e}")

def preprocess_data(df, sequence_length=5, use_tf_dataset=True):
    """
    LSTM 모델을 위한 시퀀스 데이터 생성 (RTX 4060 최적화)

    Args:
        df: 로또 데이터가 포함된 DataFrame
        sequence_length: 몇 개의 이전 회차를 사용할지 결정
        use_tf_dataset: 최적화된 TF 데이터셋 사용 여부

    Returns:
        X_array: 입력 시퀀스
        y_array: 출력 레이블
        scaler: 데이터 스케일러
    """
    # 로또 번호 컬럼만 선택
    lotto_numbers = df[['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']].values

    # 데이터 정규화 (0-1 사이 값으로)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_numbers = scaler.fit_transform(lotto_numbers)

    # 시퀀스 데이터 생성
    X, y = [], []
    for i in range(len(scaled_numbers) - sequence_length):
        # 입력: 이전 sequence_length 회차의 번호들
        X.append(scaled_numbers[i:i+sequence_length])
        # 출력: 다음 회차의 번호들
        y.append(scaled_numbers[i+sequence_length])

    # FP16 혼합 정밀도 계산을 위한 float32 데이터 타입 사용
    X_array = np.array(X, dtype=np.float32)
    y_array = np.array(y, dtype=np.float32)

    # TF 데이터셋으로 변환 (선택적)
    if use_tf_dataset:
        create_optimized_dataset(X_array, y_array)

    return X_array, y_array, scaler

def create_optimized_dataset(X, y, batch_size=64, shuffle_buffer=1000):
    """
    RTX 4060에 최적화된 TensorFlow 데이터셋 생성

    Args:
        X: 입력 시퀀스 배열
        y: 출력 레이블 배열
        batch_size: 배치 크기
        shuffle_buffer: 셔플 버퍼 크기

    Returns:
        dataset: 최적화된 TF 데이터셋
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    # 데이터셋 최적화
    dataset = (dataset
              .cache()                                  # 메모리에 데이터셋 캐싱
              .shuffle(buffer_size=shuffle_buffer)      # 데이터 셔플
              .batch(batch_size)                        # 배치 처리
              .prefetch(tf.data.AUTOTUNE))              # 자동 프리페치 최적화

    return dataset

def get_latest_sequence(df, sequence_length, scaler):
    """
    가장 최근 sequence_length 회차의 데이터를 가져와 예측에 사용할 시퀀스 생성

    Args:
        df: 로또 데이터가 포함된 DataFrame
        sequence_length: 사용할 이전 회차 수
        scaler: 학습에 사용된 스케일러

    Returns:
        시퀀스 데이터 (예측용)
    """
    latest_data = df.iloc[:sequence_length][['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']].values
    scaled_latest = scaler.transform(latest_data)

    # RTX 4060 최적화 (float32 사용)
    return np.array([scaled_latest], dtype=np.float32)