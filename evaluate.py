"""
모델 성능 평가 모듈
- 과거 데이터로 모델 정확도 테스트
- 예측 결과 분석
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# 모듈 가져오기
from data_loader import load_data, preprocess_data
from model import build_model, train_and_evaluate, predict_next_numbers
from utils import compare_predictions, calculate_win_probability

def evaluate_model_with_historical_data(file_path='lotto.xlsx', sequence_length=5, test_size=20, epochs=50):
    """과거 데이터로 모델 성능 평가"""
    print(f"\n{test_size}개의 과거 회차 데이터로 모델 성능을 평가합니다...")
    
    # 1. 데이터 로드
    df = load_data(file_path)
    
    # 테스트에 사용할 회차가 충분한지 확인
    if len(df) < test_size + sequence_length:
        print(f"충분한 데이터가 없습니다. 최소 {test_size + sequence_length}개의 회차 데이터가 필요합니다.")
        return None
    
    # 2. 평가용 데이터 준비
    results = []
    
    for test_idx in range(test_size):
        # 테스트 회차보다 이전 데이터만 사용
        train_df = df.iloc[test_idx+1:].reset_index(drop=True)
        test_row = df.iloc[test_idx]
        
        # 실제 당첨 번호
        actual_numbers = [test_row[f'번호{j}'] for j in range(1, 7)]
        
        # 데이터 전처리
        X, y, scaler = preprocess_data(train_df, sequence_length)
        
        # 학습용/검증용 데이터 비율 결정 (데이터 양에 따라 조정)
        if len(train_df) > 100:
            val_split = 0.2
        else:
            val_split = 0.1
        
        # 모델 학습
        model, _ = train_and_evaluate(X, y, epochs=epochs)
        
        # 해당 회차 예측을 위한 시퀀스 데이터
        latest_data = train_df.iloc[:sequence_length][['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']].values
        scaled_latest = scaler.transform(latest_data)
        latest_sequence = np.array([scaled_latest])
        
        # 예측
        predicted_numbers = predict_next_numbers(model, latest_sequence, scaler)
        
        # 결과 비교
        match_count, matched_numbers = compare_predictions(actual_numbers, predicted_numbers)
        
        # 결과 저장
        results.append({
            '회차': test_row['회차'],
            '실제번호': actual_numbers,
            '예측번호': predicted_numbers,
            '일치개수': match_count,
            '일치번호': matched_numbers
        })
        
        print(f"회차 {test_row['회차']} 평가: 일치 개수 {match_count}개 {matched_numbers}")
    
    # 3. 결과 분석
    results_df = pd.DataFrame(results)
    
    # 일치 개수별 통계
    match_counts = results_df['일치개수'].value_counts().sort_index()
    
    print("\n=== 모델 평가 결과 ===")
    print(f"평가한 회차 수: {test_size}")
    
    total_matches = sum(match_counts.index * match_counts.values)
    avg_matches = total_matches / test_size if test_size > 0 else 0
    
    print(f"평균 일치 개수: {avg_matches:.2f}")
    print("\n일치 개수별 분포:")
    
    for count, freq in match_counts.items():
        percentage = (freq / test_size) * 100
        print(f"{count}개 일치: {freq}회 ({percentage:.1f}%)")
    
    # 3개 이상 일치 확률
    win_prob = sum(match_counts.get(i, 0) for i in range(3, 7)) / test_size if test_size > 0 else 0
    print(f"\n3개 이상 일치 확률: {win_prob:.2f} ({win_prob*100:.1f}%)")
    
    # 그래프로 시각화
    plt.figure(figsize=(10, 6))
    ax = match_counts.plot(kind='bar', color='skyblue')
    plt.title('일치 개수별 분포', fontsize=16)
    plt.xlabel('일치한 번호 개수', fontsize=12)
    plt.ylabel('횟수', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # 바 위에 값 표시
    for i, v in enumerate(match_counts):
        ax.text(i, v + 0.1, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png')
    print("\n평가 결과 그래프가 'model_evaluation.png'로 저장되었습니다.")
    
    return results_df

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" " * 25 + "모델 성능 평가")
    print("="*70)
    
    file_path = 'lotto.xlsx'     # 데이터 파일 경로
    sequence_length = 5          # 예측에 사용할 이전 회차 수
    test_size = 20               # 평가에 사용할 과거 회차 수
    epochs = 50                  # 학습 에포크 수 (평가용이므로 적게 설정)
    
    try:
        # 모델 평가 실행
        results = evaluate_model_with_historical_data(
            file_path, 
            sequence_length, 
            test_size, 
            epochs
        )
        
        if results is not None:
            print("\n" + "="*70)
            print(" " * 20 + "모델 평가가 완료되었습니다!")
            print("="*70)
            print("""
[참고 사항]
- 이 평가 결과는 과거 데이터에 대한 모델의 성능을 보여줍니다.
- 실제 로또 추첨은 무작위이므로, 과거 성능이 미래 성능을 보장하지 않습니다.
- 일반적으로 로또 번호 3개 일치 확률은 약 1%입니다.

주의: 이 모델은 학습 및 참고용으로만 사용하시기 바랍니다.
""")
            print("="*70)
    
    except Exception as e:
        print(f"\n오류가 발생했습니다: {e}")
        print("파일 경로와 형식을 확인하고 다시 시도해주세요.")
