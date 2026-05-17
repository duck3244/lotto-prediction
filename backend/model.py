"""
LSTM 모델 정의 및 학습 모듈 (RTX 4060 최적화)
- 양방향 LSTM 모델 구축
- 배치 정규화 및 드롭아웃 적용
- 혼합 정밀도 학습 지원
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")  # 헤드리스/ASGI 환경에서 GUI 백엔드 회피
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.preprocessing import MinMaxScaler

from data_loader import create_optimized_dataset
from constants import NUM_BALLS, MIN_NUMBER, MAX_NUMBER
from paths import (
    MODELS_DIR,
    TENSORBOARD_DIR,
    VISUALIZATION_DIR,
    LEARNING_CURVE_PATH,
    DEFAULT_QUANTIZED_MODEL_BASENAME,
)


def build_model(input_shape: tuple[int, int], use_gpu: bool = True) -> Sequential:
    """
    RTX 4060에 최적화된 양방향 LSTM 모델 구축

    Args:
        input_shape: 입력 데이터 형태 (sequence_length, features)
        use_gpu: GPU 최적화 사용 여부

    Returns:
        model: 컴파일된 LSTM 모델
    """
    model = Sequential(name="LottoPredictor")

    # 첫 번째 양방향 LSTM 층
    model.add(Bidirectional(LSTM(192, return_sequences=True),
                          input_shape=input_shape,
                          name="bidirectional_lstm_1"))
    model.add(BatchNormalization(name="batch_norm_1"))
    model.add(Dropout(0.25, name="dropout_1"))

    # 두 번째 양방향 LSTM 층
    model.add(Bidirectional(LSTM(128), name="bidirectional_lstm_2"))
    model.add(BatchNormalization(name="batch_norm_2"))
    model.add(Dropout(0.25, name="dropout_2"))

    # 완전 연결 층
    model.add(Dense(96, activation='relu', name="dense_1"))
    model.add(BatchNormalization(name="batch_norm_3"))
    model.add(Dropout(0.2, name="dropout_3"))

    # 출력층 (로또 번호 개수) — mixed_float16 정책에서도 수치 안정성을 위해 float32 고정
    model.add(Dense(NUM_BALLS, name="output", dtype='float32'))

    # GPU 최적화 컴파일 설정
    if use_gpu:
        # RTX 4060에 최적화된 Adam 설정
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
    else:
        optimizer = 'adam'

    model.compile(optimizer=optimizer, loss='mse')

    return model


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 300,
    batch_size: int = 64,
    validation_split: float = 0.2,
    use_gpu: bool = True,
    extra_callbacks: list | None = None,
) -> tuple[Sequential, tf.keras.callbacks.History]:
    """
    RTX 4060에 최적화된 모델 학습 및 평가

    Args:
        X: 입력 시퀀스 데이터
        y: 목표(타겟) 데이터
        epochs: 최대 학습 에포크 수
        batch_size: 배치 크기
        validation_split: 검증 데이터 비율
        use_gpu: GPU 최적화 사용 여부

    Returns:
        model: 학습된 모델
        history: 학습 이력
    """
    # 모델 저장 디렉토리 (절대경로 — backend/paths.py 기준)
    MODELS_DIR.mkdir(exist_ok=True)
    TENSORBOARD_DIR.mkdir(exist_ok=True, parents=True)

    # 데이터 분할 (시계열: 앞쪽=과거=학습, 뒤쪽=최신=검증)
    val_size = int(len(X) * validation_split)
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]

    # TF Dataset 파이프라인 구축 (학습 데이터만 shuffle, 검증 데이터는 순서 유지)
    train_dataset = create_optimized_dataset(X_train, y_train, batch_size=batch_size,
                                              shuffle_buffer=len(X_train))
    val_dataset = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
                   .batch(batch_size)
                   .cache()
                   .prefetch(tf.data.AUTOTUNE))

    # 모델 구축
    model = build_model((X.shape[1], X.shape[2]), use_gpu)

    # 콜백 함수 정의
    callbacks = [
        # 조기 종료 (과적합 방지)
        EarlyStopping(
            monitor='val_loss',
            patience=30,
            verbose=1,
            restore_best_weights=True
        ),
        # 학습률 감소 (정체 구간에서 성능 향상)
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            verbose=1,
            min_lr=0.00001
        ),
        # TensorBoard 로깅
        TensorBoard(
            log_dir=str(TENSORBOARD_DIR),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    if extra_callbacks:
        callbacks = callbacks + list(extra_callbacks)

    # 모델 학습 (TF Dataset 파이프라인 사용)
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )

    # 학습 과정 시각화 — fig 핸들을 직접 잡아 누수 방지
    fig = plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Learning')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 시각화 디렉토리 확인 및 생성 (절대경로)
    VISUALIZATION_DIR.mkdir(exist_ok=True)

    fig.savefig(LEARNING_CURVE_PATH)
    plt.close(fig)

    return model, history


def predict_next_numbers(
    model: Sequential, latest_sequence: np.ndarray, scaler: MinMaxScaler,
) -> list[int]:
    """
    학습된 모델로 다음 회차 번호 예측

    모델 출력의 각 위치별 예측을 보존하면서 중복 시 가장 가까운 미사용 번호로
    결정적으로 재할당한다. 무작위 패딩을 사용하지 않으므로 동일 입력은
    동일 결과를 반환한다.
    """
    predicted_normalized = model.predict(latest_sequence, verbose=0)
    predicted_raw = scaler.inverse_transform(predicted_normalized)[0]

    used = set()
    final_numbers = []
    for raw in predicted_raw:
        candidate = int(max(MIN_NUMBER, min(MAX_NUMBER, round(float(raw)))))
        if candidate in used:
            # 양쪽으로 거리를 늘려가며 가장 가까운 미사용 번호 탐색
            replacement = None
            for offset in range(1, MAX_NUMBER - MIN_NUMBER + 1):
                for delta in (offset, -offset):
                    cand = candidate + delta
                    if MIN_NUMBER <= cand <= MAX_NUMBER and cand not in used:
                        replacement = cand
                        break
                if replacement is not None:
                    break
            candidate = replacement
        used.add(candidate)
        final_numbers.append(candidate)

    return sorted(final_numbers)


def export_quantized_model(model: Sequential, output_path: str | None = None) -> str:
    """
    학습된 모델을 양자화하여 TF Lite 형식으로 저장 (크기 감소 및 추론 가속)

    Args:
        model: 학습된 Keras 모델
        output_path: 저장할 파일 경로

    Returns:
        tflite_path: 저장된 TF Lite 모델 경로
    """
    if output_path is None:
        output_path = str(DEFAULT_QUANTIZED_MODEL_BASENAME)
    MODELS_DIR.mkdir(exist_ok=True)

    # TF Lite 변환기 생성
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 양자화 설정
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 모델 변환
    tflite_model = converter.convert()

    # 파일로 저장
    tflite_path = f"{output_path}.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"양자화된 모델이 {tflite_path}에 저장되었습니다.")
    return tflite_path

