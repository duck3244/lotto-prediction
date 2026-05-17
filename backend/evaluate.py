"""
모델 성능 평가 모듈
- 단일 학습 + 워크포워드(walk-forward) 평가
- 랜덤 베이스라인과 비교하여 통계적 유의성 점검
"""

from __future__ import annotations

# IMPORTANT: TensorFlow 결정성 환경변수는 ``import tensorflow`` 이전에 세팅되어야 한다.
from env_setup import set_deterministic_env

set_deterministic_env()

import argparse
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 헤드리스/ASGI 환경에서 GUI 백엔드 회피
import matplotlib.pyplot as plt

from data_loader import load_data, preprocess_data
from model import train_and_evaluate, predict_next_numbers
from utils import compare_predictions, set_global_seeds, setup_gpu
from constants import LOTTO_COLUMNS, NUM_BALLS, MIN_NUMBER, MAX_NUMBER
from paths import DEFAULT_DATA_FILE, EVALUATION_PLOT_PATH, VISUALIZATION_DIR


def _random_baseline_matches(actual_numbers: list[int], rng: np.random.Generator) -> int:
    """무작위 6개 추첨과 실제 번호의 일치 개수 (베이스라인)."""
    pick = rng.choice(np.arange(MIN_NUMBER, MAX_NUMBER + 1), size=NUM_BALLS, replace=False)
    return len(set(actual_numbers) & set(pick.tolist()))


def evaluate_model_with_historical_data(
    file_path: str | None = None,
    sequence_length: int = 10,
    test_size: int = 20,
    epochs: int = 50,
    seed: int = 42,
) -> Optional[pd.DataFrame]:
    """
    과거 데이터로 모델 성능 평가 (walk-forward)

    이전 구현은 평가 대상 회차마다 모델을 처음부터 다시 학습시켜 매우 비효율적이었다.
    여기서는 가장 최근 ``test_size``개 회차를 홀드아웃하고 나머지로 단 한 번만 학습한 뒤,
    각 테스트 회차에 대해 그 직전 ``sequence_length`` 회차 시퀀스로 예측해 비교한다.
    """
    print(f"\n{test_size}개의 과거 회차 데이터로 모델 성능을 평가합니다...")

    if file_path is None:
        file_path = str(DEFAULT_DATA_FILE)

    set_global_seeds(seed)
    setup_gpu()

    df = load_data(file_path)
    if len(df) < test_size + sequence_length + 1:
        print(f"충분한 데이터가 없습니다. 최소 {test_size + sequence_length + 1}개의 회차가 필요합니다.")
        return None

    # df는 내림차순(최신→과거). 최신 test_size개를 홀드아웃으로 분리
    test_df = df.iloc[:test_size].reset_index(drop=True)
    train_df = df.iloc[test_size:].reset_index(drop=True)

    print(f"학습 회차 수: {len(train_df)}, 평가 회차 수: {len(test_df)}")

    # 단일 학습 (스케일러는 학습 데이터로만 fit)
    X, y, scaler = preprocess_data(train_df, sequence_length)
    print(f"학습 시퀀스: X={X.shape}, y={y.shape}")
    model, _ = train_and_evaluate(X, y, epochs=epochs)

    # 워크포워드 예측 — 각 테스트 회차 i에 대해 그 직전 sequence_length 회차 시퀀스 사용
    # (df가 내림차순이므로 인덱스 i의 직전 시퀀스는 i+1..i+sequence_length)
    rng = np.random.default_rng(seed)
    results = []
    for i in range(test_size - 1, -1, -1):  # 과거→최신 순으로 평가
        test_row = test_df.iloc[i]
        actual_numbers = [int(test_row[col]) for col in LOTTO_COLUMNS]

        # 직전 sequence_length 회차 (테스트 회차 i 이후 인덱스 = 더 과거)
        prev_block = df.iloc[i + 1 : i + 1 + sequence_length][LOTTO_COLUMNS].values
        # 학습 시 사용한 정렬(과거→최신)에 맞춰 뒤집기
        prev_block = prev_block[::-1]
        scaled_block = scaler.transform(prev_block)
        latest_sequence = np.array([scaled_block], dtype=np.float32)

        predicted_numbers = predict_next_numbers(model, latest_sequence, scaler)
        match_count, matched_numbers = compare_predictions(actual_numbers, predicted_numbers)
        baseline = _random_baseline_matches(actual_numbers, rng)

        results.append(
            {
                "회차": int(test_row["회차"]),
                "실제번호": actual_numbers,
                "예측번호": predicted_numbers,
                "일치개수": match_count,
                "일치번호": matched_numbers,
                "랜덤_일치개수": baseline,
            }
        )
        print(
            f"회차 {int(test_row['회차'])} 평가: 모델 {match_count}개 일치 {matched_numbers} | "
            f"랜덤 {baseline}개"
        )

    results_df = pd.DataFrame(results)
    match_counts = results_df["일치개수"].value_counts().sort_index()

    print("\n=== 모델 평가 결과 ===")
    print(f"평가한 회차 수: {test_size}")
    avg_model = results_df["일치개수"].mean()
    avg_random = results_df["랜덤_일치개수"].mean()
    print(f"평균 일치 개수 (모델): {avg_model:.2f}")
    print(f"평균 일치 개수 (랜덤 베이스라인): {avg_random:.2f}")

    print("\n일치 개수별 분포:")
    for count, freq in match_counts.items():
        percentage = (freq / test_size) * 100
        print(f"  {count}개 일치: {freq}회 ({percentage:.1f}%)")

    win_prob = sum(match_counts.get(i, 0) for i in range(3, NUM_BALLS + 1)) / test_size
    print(f"\n3개 이상 일치 확률: {win_prob:.2f} ({win_prob * 100:.1f}%)")

    # 모델 vs 랜덤 비교 그래프
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.arange(0, NUM_BALLS + 2) - 0.5
    ax.hist(
        [results_df["일치개수"], results_df["랜덤_일치개수"]],
        bins=bins,
        label=["Model", "Random baseline"],
        color=["steelblue", "lightcoral"],
        rwidth=0.85,
    )
    ax.set_title("Match Count Distribution: Model vs Random", fontsize=15)
    ax.set_xlabel("Number of Matches")
    ax.set_ylabel("Count")
    ax.set_xticks(range(0, NUM_BALLS + 1))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    VISUALIZATION_DIR.mkdir(exist_ok=True)
    fig.savefig(EVALUATION_PLOT_PATH)
    plt.close(fig)
    print(f"\n평가 결과 그래프가 '{EVALUATION_PLOT_PATH}'로 저장되었습니다.")

    return results_df


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="로또 LSTM 모델 성능 평가 (walk-forward)")
    parser.add_argument("--file", type=str, default=str(DEFAULT_DATA_FILE), help="데이터 파일 경로")
    parser.add_argument("--sequence", type=int, default=10, help="시퀀스 길이")
    parser.add_argument("--test-size", type=int, default=20, help="홀드아웃 평가 회차 수")
    parser.add_argument("--epochs", type=int, default=50, help="학습 에포크 수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    return parser.parse_args()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" " * 25 + "모델 성능 평가")
    print("=" * 70)

    args = parse_arguments()

    try:
        results = evaluate_model_with_historical_data(
            file_path=args.file,
            sequence_length=args.sequence,
            test_size=args.test_size,
            epochs=args.epochs,
            seed=args.seed,
        )

        if results is not None:
            print("\n" + "=" * 70)
            print(" " * 20 + "모델 평가가 완료되었습니다!")
            print("=" * 70)
            print(
                """
[참고 사항]
- 이 평가 결과는 과거 데이터에 대한 모델의 성능을 보여줍니다.
- 실제 로또 추첨은 무작위이므로, 과거 성능이 미래 성능을 보장하지 않습니다.
- 일반적으로 로또 번호 3개 일치 확률은 약 1.8%입니다 (이론값).
- 랜덤 베이스라인과 비교해 유의미하게 더 높지 않다면 모델의 예측력이 없는 것으로
  보아야 합니다.

주의: 이 모델은 학습 및 참고용으로만 사용하시기 바랍니다.
"""
            )
            print("=" * 70)

    except Exception as e:
        print(f"\n오류가 발생했습니다: {e}")
        print("파일 경로와 형식을 확인하고 다시 시도해주세요.")
