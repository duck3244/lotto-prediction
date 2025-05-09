#!/usr/bin/env python3
"""
RTX 4060에 최적화된 로또 번호 예측 시스템

이 스크립트는 LSTM 기반 로또 번호 예측 시스템의 메인 실행 파일입니다.
전체 학습 및 예측 프로세스를 관리하며 RTX 4060 GPU에 최적화되어 있습니다.
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import tensorflow as tf

from pathlib import Path

# 모듈 불러오기
from data_loader import load_data, preprocess_data, get_latest_sequence
from model import build_model, train_and_evaluate, predict_next_numbers
from analysis import analyze_patterns, ensemble_prediction
from visualization import visualize_data
from utils import save_model, load_model, setup_gpu_monitoring

# 전역 변수
LOGS_DIR = Path("logs")
MODELS_DIR = Path("models")
VISUALIZATION_DIR = Path("visualization")


def setup_logging(log_level=logging.INFO):
    """로깅 시스템 설정"""
    LOGS_DIR.mkdir(exist_ok=True)

    log_file = LOGS_DIR / f"lotto_prediction_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def setup_gpu():
    """RTX 4060에 최적화된 GPU 설정"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # 메모리 증가 설정
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # 혼합 정밀도 계산 활성화 (FP16 사용)
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)

            # XLA 컴파일 활성화
            tf.config.optimizer.set_jit(True)

            gpu_info = tf.config.experimental.get_device_details(gpus[0])
            gpu_name = gpu_info.get('device_name', 'Unknown')

            return True, f"GPU 설정 완료: {gpu_name}, 혼합 정밀도 및 XLA 컴파일 활성화"
        except RuntimeError as e:
            return False, f"GPU 설정 오류: {e}"
    return False, "GPU를 찾을 수 없습니다. CPU 모드로 실행됩니다."


def create_directories():
    """필요한 디렉토리 생성"""
    dirs = [LOGS_DIR, MODELS_DIR, VISUALIZATION_DIR]
    for dir_path in dirs:
        dir_path.mkdir(exist_ok=True)
    return dirs


def show_help():
    """사용자를 위한 도움말과 모델 설명"""
    print("\n" + "=" * 70)
    print(" " * 25 + "LSTM 로또 번호 예측 모델")
    print("=" * 70)
    print("""
이 프로그램은 딥러닝 LSTM(Long Short-Term Memory) 모델을 사용하여 로또 번호를
예측합니다. RTX 4060 GPU에 최적화되어 있으며, 과거 당첨 번호 데이터를 분석하여
패턴을 학습하고 다음 회차에 나올 가능성이 높은 번호를 예측합니다.

[주요 기능]
1. RTX 4060에 최적화된 양방향 LSTM 모델 사용
2. 혼합 정밀도 학습 및 XLA 컴파일로 성능 향상
3. 번호 출현 빈도, 홀짝 분포, 범위별 분포 등 다양한 통계 분석
4. 다양한 예측 방법을 결합한 앙상블 예측
5. 데이터 시각화 및 분석 결과 시각화

[중요 참고 사항]
* 이 모델은 학습 목적으로 만들어진 것으로, 실제 당첨을 보장하지 않습니다.
* 로또는 본질적으로 무작위 추첨 방식이므로, 과거 데이터에서 의미 있는 패턴을
  찾기 어려울 수 있습니다.
* 모델의 예측 결과는 참고용으로만 사용하시기 바랍니다.
* 책임감 있는 복권 구매를 권장합니다.

[명령행 인수]
--file FILE         로또 데이터 파일 경로 (기본값: lotto.xlsx)
--sequence N        예측에 사용할 이전 회차 수 (기본값: 10)
--epochs N          최대 학습 에포크 수 (기본값: 300)
--batch-size N      배치 크기 (기본값: 64)
--no-visualize      시각화 생성 비활성화
--verbose           상세 로그 출력
--load-model PATH   저장된 모델 로드
""")
    print("=" * 70)


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="RTX 4060에 최적화된 LSTM 로또 예측 시스템")

    parser.add_argument("--file", type=str, default="lotto.xlsx",
                        help="로또 데이터 엑셀 파일 경로 (기본값: lotto.xlsx)")

    parser.add_argument("--sequence", type=int, default=10,
                        help="예측에 사용할 이전 회차 수 (기본값: 10)")

    parser.add_argument("--epochs", type=int, default=300,
                        help="최대 학습 에포크 수 (기본값: 300)")

    parser.add_argument("--batch-size", type=int, default=64,
                        help="배치 크기 (기본값: 64)")

    parser.add_argument("--visualize", dest="visualize", action="store_true",
                        help="데이터 시각화 생성 (기본값)")

    parser.add_argument("--no-visualize", dest="visualize", action="store_false",
                        help="데이터 시각화 생성 안 함")

    parser.add_argument("--verbose", "-v", action="store_true",
                        help="상세 로그 출력")

    parser.add_argument("--load-model", type=str, default=None,
                        help="저장된 모델 파일 경로")

    parser.add_argument("--monitor-gpu", action="store_true",
                        help="GPU 사용량 모니터링 활성화")

    parser.set_defaults(visualize=True)

    return parser.parse_args()


def main(args, logger):
    """Main execution function"""
    # Record start time
    start_time = time.time()

    # GPU setup
    gpu_available, gpu_message = setup_gpu()
    logger.info(gpu_message)

    # GPU monitoring setup (optional)
    monitor_stop_func = None
    if args.monitor_gpu and gpu_available:
        _, monitor_stop_func = setup_gpu_monitoring(interval=5)

    try:
        # 1. Load data
        logger.info(f"Loading data file '{args.file}'...")
        df = load_data(args.file)
        logger.info(f"Lottery data loaded: {len(df)} draws")

        # 2. Preprocess data
        logger.info(f"Preprocessing data with sequence length {args.sequence}...")
        X, y, scaler = preprocess_data(df, args.sequence, use_tf_dataset=True)
        logger.info(f"Data preprocessing complete: X shape {X.shape}, y shape {y.shape}")

        # 3. Build or load model
        if args.load_model:
            logger.info(f"Loading model from '{args.load_model}'...")
            model = load_model(args.load_model)
            if model is None:
                logger.error("Failed to load model. Training new model.")
                model, history = train_and_evaluate(
                    X, y, epochs=args.epochs, batch_size=args.batch_size, use_gpu=gpu_available
                )
            else:
                logger.info("Model loaded successfully")
                history = None
        else:
            # Train model
            logger.info(
                f"Starting RTX 4060 optimized model training (epochs: {args.epochs}, batch size: {args.batch_size})...")
            model_start_time = time.time()
            model, history = train_and_evaluate(
                X, y, epochs=args.epochs, batch_size=args.batch_size, use_gpu=gpu_available
            )
            model_time = time.time() - model_start_time
            logger.info(f"Model training completed: {model_time:.2f} seconds")

        # 4. Predict next draw
        logger.info("Predicting next draw numbers...")
        latest_sequence = get_latest_sequence(df, args.sequence, scaler)

        prediction_start_time = time.time()
        lstm_prediction = predict_next_numbers(model, latest_sequence, scaler)
        prediction_time = time.time() - prediction_start_time

        logger.info(f"LSTM prediction complete ({prediction_time:.2f} seconds): {lstm_prediction}")

        # 5. Pattern analysis and ensemble prediction
        logger.info("Analyzing lottery number patterns...")
        frequencies, odd_even_patterns, range_patterns = analyze_patterns(df)

        logger.info("Generating ensemble prediction...")
        ensemble_result = ensemble_prediction(df, lstm_prediction, frequencies, args.sequence)
        logger.info(f"Ensemble prediction: {ensemble_result}")

        # 6. Visualization (optional)
        if args.visualize:
            logger.info("Generating data visualizations...")
            visualize_data(df, frequencies, VISUALIZATION_DIR)
            logger.info(f"Visualization complete: results saved in {VISUALIZATION_DIR} directory")

        # 7. Save model
        if not args.load_model or (args.load_model and history is not None):
            model_filename = f"lotto_lstm_model_{time.strftime('%Y%m%d_%H%M%S')}.h5"
            model_path = MODELS_DIR / model_filename
            save_model(model, str(model_path))
            logger.info(f"Model saved: {model_path}")

        # 8. Calculate execution time
        total_time = time.time() - start_time
        logger.info(f"Total execution time: {total_time:.2f} seconds")

        # 9. Return results
        return model, lstm_prediction, ensemble_result

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\nError: File '{args.file}' not found or cannot be accessed.")
        return None
    except Exception as e:
        logger.exception(f"Exception in main function: {e}")
        print(f"\nError occurred: {e}")
        return None
    finally:
        # Stop GPU monitoring if started
        if monitor_stop_func:
            monitor_stop_func()


if __name__ == "__main__":
    # 명령행 인수 파싱
    args = parse_arguments()

    # 디렉토리 생성
    create_directories()

    # 로깅 설정
    logger = setup_logging(log_level=logging.DEBUG if args.verbose else logging.INFO)
    logger.info("로또 예측 시스템 시작")

    # 도움말 표시
    show_help()

    print("\n프로그램을 실행합니다...\n")

    try:
        # 모델 실행
        result = main(args, logger)  # 결과를 단일 변수로 받음

        # None 체크 추가
        if result is None:
            print("\n오류: 모델 실행에 실패했습니다. 로그 파일을 확인하세요.")
            sys.exit(1)

        model, lstm_prediction, ensemble_prediction = result  # 성공한 경우에만 언패킹

        if lstm_prediction is not None and ensemble_prediction is not None:
            print("\n" + "=" * 70)
            print(" " * 20 + "Program completed successfully!")
            print("=" * 70)
            print(f"""
[Final Prediction Numbers]
1. LSTM Model Prediction: {lstm_prediction}
2. Ensemble Model Prediction: {ensemble_prediction}

[Next Steps]
- You can choose numbers based on the model results.
- Models are saved in the 'models' directory for future use.
- Visualizations are saved in the 'visualization' directory.

Good luck! We recommend responsible lottery participation.
""")
            print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 프로그램이 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n오류가 발생했습니다: {e}")
        logger.exception("Unhandled exception in main program")
        sys.exit(1)

