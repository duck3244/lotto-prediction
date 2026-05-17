"""
학습된 모델을 사용하여 로또 번호를 예측하는 모듈 (RTX 4060 최적화)
- 저장된 모델 로드
- 새로운 번호 예측
- 다양한 예측 전략 제공
"""

# IMPORTANT: TensorFlow 결정성 환경변수는 ``import tensorflow`` 이전에 세팅되어야 한다.
from env_setup import set_deterministic_env

set_deterministic_env()

import os
import sys
import argparse
import numpy as np
import tensorflow as tf

# 로컬 모듈 가져오기
from data_loader import load_data, get_latest_sequence
from model import predict_next_numbers
from analysis import analyze_patterns, ensemble_prediction
from utils import (
    load_model, load_scaler, validate_lotto_numbers, suggest_balanced_numbers,
    setup_gpu, setup_gpu_monitoring, set_global_seeds, load_training_bundle,
)
from visualization import visualize_data, visualize_prediction_comparison
from constants import LOTTO_COLUMNS
from paths import DEFAULT_DATA_FILE, VISUALIZATION_DIR


def predict_with_saved_model(args):
    """
    저장된 모델을 사용하여 로또 번호 예측

    Args:
        args: 명령행 인수

    Returns:
        lstm_prediction: LSTM 모델 예측 결과
        ensemble_prediction: 앙상블 예측 결과
        additional_sets: 추가 추천 번호 세트
    """
    print(f"\n{'-'*30} 로또 번호 예측 시작 {'-'*30}")

    # 0. Set global random seeds
    set_global_seeds(args.seed)

    # 1. GPU 설정 최적화
    if args.gpu:
        success, message = setup_gpu()
        print(message)

        if args.monitor_gpu:
            _, stop_monitoring = setup_gpu_monitoring(interval=5)

    # 2. 모델 + 스케일러 로드 — 번들 경로가 주어지면 번들에서, 아니면 레거시 모드
    if args.bundle:
        print(f"\n학습 번들 '{args.bundle}' 로드 중...")
        try:
            model, scaler, meta = load_training_bundle(args.bundle)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"번들 로드 실패: {e}")
            return None, None, None
        print(f"번들 로드 성공 (학습 시점: {meta.get('timestamp')}, "
              f"sequence_length={meta.get('sequence_length')}, "
              f"data_sha256={(meta.get('data_sha256') or '')[:12]}…)")
    else:
        # 레거시: --model + --scaler 페어를 사용자가 직접 매칭
        if not args.model:
            print("오류: --bundle 또는 --model+--scaler 중 하나는 반드시 지정해야 합니다.")
            return None, None, None
        if not args.scaler:
            print("오류: --scaler 인자로 학습 시 저장된 스케일러 파일을 지정해야 합니다.")
            print("       (현재 데이터로 새 스케일러를 만들면 학습과 데이터 분포가 달라 예측이 부정확합니다.)")
            return None, None, None

        print(f"\n모델 '{args.model}'에서 로드 중...")
        model = load_model(args.model)
        if model is None:
            print("모델을 불러올 수 없습니다. 프로그램을 종료합니다.")
            return None, None, None
        print("모델 로드 성공")

        scaler = load_scaler(args.scaler)
        if scaler is None:
            print(f"오류: 스케일러 파일을 로드할 수 없습니다: {args.scaler}")
            return None, None, None
        print(f"스케일러 로드 성공: {args.scaler}")

    # 3. 데이터 로드
    print(f"\n데이터 파일 '{args.file}'에서 로또 데이터 로드 중...")
    try:
        df = load_data(args.file)
        print(f"데이터 로드 성공: {len(df)}개 회차")
    except Exception as e:
        print(f"데이터 로드 오류: {e}")
        return None, None, None

    # 최근 시퀀스 데이터 가져오기
    latest_sequence = get_latest_sequence(df, args.sequence, scaler)

    # 5. LSTM 모델 예측
    print("\nLSTM 모델로 예측 중...")
    lstm_prediction = predict_next_numbers(model, latest_sequence, scaler)
    print(f"LSTM 모델 예측 번호: {lstm_prediction}")

    # 번호 유효성 검사
    is_valid, message = validate_lotto_numbers(lstm_prediction)
    if not is_valid:
        print(f"경고: {message}")

    # 6. 추가 분석 및 앙상블 예측
    print("\n패턴 분석 중...")
    frequencies, odd_even_patterns, range_patterns = analyze_patterns(df)

    print("\n앙상블 예측 생성 중...")
    ensemble_result = ensemble_prediction(df, lstm_prediction, frequencies, args.sequence)
    print(f"앙상블 예측 번호: {ensemble_result}")

    # 7. 추가 추천 번호 세트 생성
    print("\n추가 추천 번호 세트 생성 중...")

    # 최근 번호 추출
    recent_numbers = set()
    for i in range(min(5, len(df))):
        for col in LOTTO_COLUMNS:
            recent_numbers.add(df.iloc[i][col])

    additional_sets = suggest_balanced_numbers(frequencies, recent_numbers, args.num_sets)

    print("\n=== 추가 추천 번호 세트 ===")
    for i, number_set in enumerate(additional_sets, 1):
        print(f"추천 세트 {i}: {number_set}")

    # 8. 시각화 (선택적)
    if args.visualize:
        print("\n데이터 시각화 생성 중...")
        VISUALIZATION_DIR.mkdir(exist_ok=True)

        visualize_data(df, frequencies, VISUALIZATION_DIR)
        visualize_prediction_comparison(lstm_prediction, ensemble_result, VISUALIZATION_DIR)

        print(f"시각화 파일이 '{VISUALIZATION_DIR}' 디렉토리에 저장되었습니다.")

    # 9. 최종 결과 요약
    print(f"\n{'-'*30} 예측 결과 요약 {'-'*30}")
    print(f"1. LSTM 모델 예측: {lstm_prediction}")
    print(f"2. 앙상블 예측: {ensemble_result}")
    print(f"3. 추가 추천 세트 ({args.num_sets}개):")
    for i, number_set in enumerate(additional_sets, 1):
        print(f"   세트 {i}: {number_set}")

    # GPU 모니터링 중지
    if args.gpu and args.monitor_gpu and 'stop_monitoring' in locals():
        stop_monitoring()

    return lstm_prediction, ensemble_result, additional_sets


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="RTX 4060에 최적화된 로또 번호 예측")

    parser.add_argument("--bundle", type=str, default=None,
                        help="학습 번들 디렉토리 경로 (모델+스케일러+메타 일체). "
                             "지정 시 --model/--scaler 보다 우선한다.")

    parser.add_argument("--model", type=str, default=None,
                        help="(레거시) 학습된 모델 파일 경로. --bundle 미지정 시 필요.")

    parser.add_argument("--scaler", type=str, default=None,
                        help="(레거시) 학습 시 저장된 스케일러 파일 경로 (.pkl). "
                             "--bundle 미지정 시 필요.")

    parser.add_argument("--file", type=str, default=str(DEFAULT_DATA_FILE),
                        help=f"로또 데이터 엑셀 파일 경로 (기본값: {DEFAULT_DATA_FILE})")

    parser.add_argument("--sequence", type=int, default=10,
                        help="예측에 사용할 이전 회차 수 (기본값: 10)")

    parser.add_argument("--num-sets", type=int, default=3,
                        help="생성할 추가 추천 세트 수 (기본값: 3)")

    parser.add_argument("--visualize", action="store_true",
                        help="데이터 시각화 생성")

    parser.add_argument("--gpu", dest="gpu", action="store_true",
                        help="GPU 최적화 활성화 (기본값)")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false",
                        help="GPU 최적화 비활성화 (CPU만 사용)")
    parser.set_defaults(gpu=True)

    parser.add_argument("--monitor-gpu", action="store_true",
                        help="GPU 사용량 모니터링 활성화")

    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드 값 (기본값: 42)")

    return parser.parse_args()


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" " * 20 + "RTX 4060에 최적화된 로또 번호 예측")
    print("="*70)

    # 명령행 인수 파싱
    args = parse_arguments()

    # 입력 경로 사전 검증 — --bundle 우선, 없으면 --model/--scaler 확인
    if args.bundle:
        if not os.path.isdir(args.bundle):
            print(f"학습 번들 디렉토리 '{args.bundle}'를 찾을 수 없습니다.")
            sys.exit(1)
    else:
        if not args.model:
            print("오류: --bundle 또는 --model 중 하나는 반드시 지정해야 합니다.")
            sys.exit(1)
        if not os.path.exists(args.model):
            print(f"모델 파일 '{args.model}'을 찾을 수 없습니다.")
            sys.exit(1)

    try:
        # 예측 실행
        lstm_prediction, ensemble_result, additional_sets = predict_with_saved_model(args)

        if lstm_prediction is not None:
            print("\n" + "="*70)
            print(" " * 20 + "예측이 완료되었습니다!")
            print("="*70)
            print("""
[참고 사항]
- 위 예측 결과는 학습된 모델에 기반한 것으로 실제 당첨을 보장하지 않습니다.
- 로또는 무작위 추첨 방식이므로 어떤 번호든 동일한 확률로 당첨될 수 있습니다.
- 이 모델은 학습 및 참고용으로만 사용하시기 바랍니다.

행운을 빕니다!
""")
            print("="*70)

    except KeyboardInterrupt:
        print("\n\n사용자에 의해 프로그램이 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n오류가 발생했습니다: {e}")
        sys.exit(1)

