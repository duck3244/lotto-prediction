"""
로또 번호 패턴 분석 모듈
- 번호 출현 빈도 분석
- 홀짝 분포 분석
- 번호 범위 패턴 분석
- 앙상블 예측 알고리즘
"""

import numpy as np

from constants import LOTTO_COLUMNS, NUM_BALLS, MIN_NUMBER, MAX_NUMBER


def analyze_patterns(df):
    """
    Analyze various lottery number patterns

    Args:
        df: Lottery data DataFrame

    Returns:
        frequencies: Number frequency dictionary
        odd_even_freq: Odd/Even combination frequency
        range_patterns: Number range pattern frequency
    """
    print("\n=== Lottery Number Pattern Analysis ===")

    # Calculate number frequencies
    frequencies = {}
    for i in range(MIN_NUMBER, MAX_NUMBER + 1):
        col_counts = [(df[col] == i).sum() for col in LOTTO_COLUMNS]
        frequencies[i] = sum(col_counts)

    # Top/bottom 10 frequency numbers
    top_numbers = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
    bottom_numbers = sorted(frequencies.items(), key=lambda x: x[1])[:10]

    print("\n- Most Frequent Numbers (Top 10):")
    for num, freq in top_numbers:
        print(f"  Number {num}: {freq} times")

    print("\n- Least Frequent Numbers (Bottom 10):")
    for num, freq in bottom_numbers:
        print(f"  Number {num}: {freq} times")

    # 벡터화 연산을 위한 번호 행렬
    numbers_matrix = df[LOTTO_COLUMNS].values

    # Count draws with consecutive numbers (벡터화)
    sorted_matrix = np.sort(numbers_matrix, axis=1)
    diffs = np.diff(sorted_matrix, axis=1)
    consecutive_count = int(np.any(diffs == 1, axis=1).sum())

    print(f"\n- Draws with consecutive numbers: {consecutive_count} ({consecutive_count / len(df) * 100:.1f}%)")

    # Odd/Even distribution (벡터화)
    odd_counts = np.sum(numbers_matrix % 2 == 1, axis=1)
    even_counts = NUM_BALLS - odd_counts

    # Odd/Even combination frequency
    odd_even_freq = {}
    for odd, even in zip(odd_counts, even_counts):
        key = f"Odd{odd}:Even{even}"
        odd_even_freq[key] = odd_even_freq.get(key, 0) + 1

    print("\n- Odd/Even Combination Distribution:")
    for combo, freq in sorted(odd_even_freq.items(), key=lambda x: x[1], reverse=True):
        print(f"  {combo}: {freq} times ({freq / len(df) * 100:.1f}%)")

    # Number range distribution (벡터화: np.digitize 사용)
    bins = [0, 10, 20, 30, 40, MAX_NUMBER]
    digitized = np.digitize(numbers_matrix, bins, right=True)  # 1~5 범위 인덱스
    range_distribution = []
    for row_bins in digitized:
        counts = tuple(np.bincount(row_bins, minlength=6)[1:])  # 인덱스 1~5
        range_distribution.append(counts)

    # Most frequent number range patterns
    range_patterns = {}
    for pattern in range_distribution:
        pattern_str = ':'.join(str(p) for p in pattern)
        range_patterns[pattern_str] = range_patterns.get(pattern_str, 0) + 1

    print("\n- Top 5 Number Range Patterns (1-10:11-20:21-30:31-40:41-45):")
    for pattern, freq in sorted(range_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {pattern}: {freq} times ({freq / len(df) * 100:.1f}%)")

    return frequencies, odd_even_freq, range_patterns


def ensemble_prediction(df, lstm_prediction, top_frequencies, sequence_length=5):
    """
    Ensemble prediction combining multiple methods (RTX 4060 optimized)

    Args:
        df: Lottery data DataFrame
        lstm_prediction: LSTM model prediction result
        top_frequencies: Number frequency dictionary
        sequence_length: Previous draw count to use

    Returns:
        ensemble_result: Final ensemble prediction numbers
    """
    print("\n=== Ensemble Prediction Methods ===")

    predictions = []
    weights = []

    # 1. LSTM model prediction (highest weight)
    predictions.append(lstm_prediction)
    weights.append(0.4)  # 40% weight
    print("1. LSTM Model Prediction:", lstm_prediction)

    # 2. Frequency-based prediction (most frequent numbers)
    frequency_based = [num for num, _ in sorted(top_frequencies.items(), key=lambda x: x[1], reverse=True)[:15]]
    np.random.shuffle(frequency_based)
    frequency_prediction = sorted(frequency_based[:6])
    predictions.append(frequency_prediction)
    weights.append(0.2)  # 20% weight
    print("2. Frequency-Based Prediction:", frequency_prediction)

    # 3. Recent pattern-based prediction (rising numbers)
    rising_numbers, _ = analyze_number_trends(df, window_size=sequence_length)
    if len(rising_numbers) >= 6:
        recent_prediction = sorted(rising_numbers[:6])
    else:
        # 상승 추세 번호가 부족하면 빈도 높은 번호로 보충
        supplement = [num for num, _ in sorted(top_frequencies.items(), key=lambda x: x[1], reverse=True)
                      if num not in rising_numbers]
        needed = 6 - len(rising_numbers)
        recent_prediction = sorted(rising_numbers + supplement[:needed])
    predictions.append(recent_prediction)
    weights.append(0.2)  # 20% weight
    print("3. Recent Pattern Prediction:", recent_prediction)

    # 4. Statistical balance prediction (odd/even and range balance)
    balanced_prediction = generate_balanced_prediction()
    predictions.append(balanced_prediction)
    weights.append(0.2)  # 20% weight
    print("4. Statistical Balance Prediction:", balanced_prediction)

    # 5. Final ensemble prediction: weighted voting
    number_votes = {}
    for i, pred_set in enumerate(predictions):
        for num in pred_set:
            if num not in number_votes:
                number_votes[num] = 0
            number_votes[num] += weights[i]

    # Select top 6 numbers by votes
    ensemble_result = sorted([num for num, _ in sorted(number_votes.items(), key=lambda x: x[1], reverse=True)[:6]])
    print("\n=== Final Ensemble Prediction Numbers ===")
    print(ensemble_result)

    return ensemble_result


def generate_balanced_prediction():
    """
    통계적으로 균형 잡힌 로또 번호 생성
    - 홀수/짝수 균형
    - 번호 범위 균형

    Returns:
        balanced_numbers: 균형 잡힌 6개 번호
    """
    # 홀수 3개, 짝수 3개 선택
    odd_numbers = [n for n in range(MIN_NUMBER, MAX_NUMBER + 1) if n % 2 == 1]
    even_numbers = [n for n in range(MIN_NUMBER, MAX_NUMBER + 1) if n % 2 == 0]

    np.random.shuffle(odd_numbers)
    np.random.shuffle(even_numbers)

    balanced_numbers = odd_numbers[:3] + even_numbers[:3]

    # 각 범위에서 최소 1개 이상 선택되도록 조정
    ranges = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 45)]

    # 각 범위에 몇 개가 선택되었는지 확인
    range_counts = [0, 0, 0, 0, 0]
    for num in balanced_numbers:
        for i, (start, end) in enumerate(ranges):
            if start <= num <= end:
                range_counts[i] += 1
                break

    # 빈 범위가 있으면 조정
    for i, count in enumerate(range_counts):
        if count == 0:
            start, end = ranges[i]
            available = [n for n in range(start, end + 1) if n not in balanced_numbers]
            if available:
                # 하나 제거하고 새로운 번호 추가
                if balanced_numbers:
                    # 중복이 많은 범위에서 제거
                    remove_from_range = np.argmax(range_counts)
                    remove_candidates = []
                    for j, num in enumerate(balanced_numbers):
                        for start, end in [ranges[remove_from_range]]:
                            if start <= num <= end:
                                remove_candidates.append(j)
                                break

                    if remove_candidates:
                        del balanced_numbers[remove_candidates[0]]
                        balanced_numbers.append(np.random.choice(available))

                        # 범위 카운트 업데이트
                        range_counts[remove_from_range] -= 1
                        range_counts[i] += 1

    return sorted(balanced_numbers)


def analyze_number_trends(df, window_size=10):
    """
    최근 추세 분석으로 상승/하락 번호 식별

    Args:
        df: 로또 데이터가 포함된 DataFrame
        window_size: 분석할 최근 회차 수

    Returns:
        rising_numbers: 상승 추세 번호 리스트
        falling_numbers: 하락 추세 번호 리스트
    """
    # 전체 데이터에서의 번호별 출현 빈도
    total_freq = {}
    for i in range(MIN_NUMBER, MAX_NUMBER + 1):
        col_counts = [(df[col] == i).sum() for col in LOTTO_COLUMNS]
        total_freq[i] = sum(col_counts)

    # 최근 window_size 회차에서의 번호별 출현 빈도
    recent_freq = {}
    recent_df = df.head(window_size)
    for i in range(MIN_NUMBER, MAX_NUMBER + 1):
        col_counts = [(recent_df[col] == i).sum() for col in LOTTO_COLUMNS]
        recent_freq[i] = sum(col_counts)

    # 전체 대비 최근 출현 비율 계산
    rising_numbers = []
    falling_numbers = []

    total_draws = len(df)
    recent_draws = len(recent_df)

    for num in range(MIN_NUMBER, MAX_NUMBER + 1):
        # 전체 및 최근 출현 확률
        total_prob = total_freq.get(num, 0) / total_draws
        recent_prob = recent_freq.get(num, 0) / recent_draws if recent_draws > 0 else 0

        # 출현 확률 변화율
        if total_prob > 0:
            change_rate = (recent_prob - total_prob) / total_prob

            # 상승/하락 번호 분류 (20% 이상 변화)
            if change_rate >= 0.2:
                rising_numbers.append(num)
            elif change_rate <= -0.2:
                falling_numbers.append(num)

    return rising_numbers, falling_numbers

