"""
로또 예측 시스템 공통 상수 정의
"""

# 로또 번호 관련 상수
NUM_BALLS = 6
MIN_NUMBER = 1
MAX_NUMBER = 45

# 데이터 컬럼명
LOTTO_COLUMNS = [f'번호{i}' for i in range(1, NUM_BALLS + 1)]
DRAW_COLUMN = '회차'
