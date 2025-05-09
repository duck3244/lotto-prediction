"""
LSTM 모델 정의 및 학습 모듈 (RTX 4060 최적화)
- 양방향 LSTM 모델 구축
- 배치 정규화 및 드롭아웃 적용
- 혼합 정밀도 학습 지원
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard


def build_model(input_shape, use_gpu=True):
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

    # 출력층 (6개 로또 번호)
    model.add(Dense(6, name="output"))

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


def train_and_evaluate(X, y, epochs=300, batch_size=64, validation_split=0.2, use_gpu=True):
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
    # 모델 저장 디렉토리
    models_dir = Path("models")
    logs_dir = Path("logs/tensorboard")
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True, parents=True)

    # 데이터 분할
    val_size = int(len(X) * validation_split)
    X_train, X_val = X[val_size:], X[:val_size]
    y_train, y_val = y[val_size:], y[:val_size]

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
        # 최상의 모델 저장
        ModelCheckpoint(
            filepath=str(models_dir / 'best_model_checkpoint.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # TensorBoard 로깅
        TensorBoard(
            log_dir=str(logs_dir),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]

    # 모델 학습
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # 학습 과정 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Learning')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 시각화 디렉토리 확인 및 생성
    vis_dir = Path("visualization")
    vis_dir.mkdir(exist_ok=True)

    plt.savefig(vis_dir / 'learning_curve.png')

    return model, history


def predict_next_numbers(model, latest_sequence, scaler):
    """
    학습된 모델로 다음 회차 번호 예측

    Args:
        model: 학습된 LSTM 모델
        latest_sequence: 최근 회차 시퀀스 데이터
        scaler: 데이터 정규화에 사용된 스케일러

    Returns:
        predicted_numbers: 예측된 6개의 로또 번호
    """
    # 모델 예측 (정규화된 값)
    predicted_normalized = model.predict(latest_sequence, verbose=0)

    # 정규화 해제
    predicted_numbers = scaler.inverse_transform(predicted_normalized)

    # 1~45 사이의 정수로 변환하고 중복 제거
    unique_numbers = set()
    for numbers in predicted_numbers:
        sorted_numbers = []
        for num in numbers:
            # 1~45 범위로 제한하고 반올림
            rounded_num = max(1, min(45, round(num)))
            sorted_numbers.append(rounded_num)

        # 중복 제거된 번호 세트 만들기
        for num in sorted_numbers:
            unique_numbers.add(num)

        # 6개 번호 선택
        if len(unique_numbers) >= 6:
            final_numbers = sorted(list(unique_numbers)[:6])
        else:
            # 번호가 부족하면 1~45에서 랜덤하게 추가
            available_numbers = [n for n in range(1, 46) if n not in unique_numbers]
            needed = 6 - len(unique_numbers)
            additional = np.random.choice(available_numbers, needed, replace=False)
            final_numbers = sorted(list(unique_numbers) + list(additional))

    return final_numbers


def export_quantized_model(model, output_path='models/quantized_model'):
    """
    학습된 모델을 양자화하여 TF Lite 형식으로 저장 (크기 감소 및 추론 가속)

    Args:
        model: 학습된 Keras 모델
        output_path: 저장할 파일 경로

    Returns:
        tflite_path: 저장된 TF Lite 모델 경로
    """
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

