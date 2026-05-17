# 아키텍처 (Architecture)

> RTX 4060 최적화 LSTM 기반 로또 번호 예측 시스템 — 단일 사용자 MVP

## 1. 개요

본 프로젝트는 과거 회차 데이터를 학습한 **LSTM 모델 + 통계 기반 앙상블**로 다음 회차 번호를 예측하는 도구입니다. 시스템은 크게 세 계층으로 구성됩니다.

- **ML 코어 (Python)** — 데이터 로딩 · 모델 학습/추론 · 분석/시각화. CLI(`main.py`, `predict.py`, `evaluate.py`)와 FastAPI 서비스가 동일한 모듈을 공유합니다.
- **API 서버 (FastAPI)** — `backend/app/`. 활성 모델/데이터 상태를 보관하고 REST 엔드포인트를 노출합니다.
- **SPA (Vue 3)** — `frontend/`. Vite + Tailwind + Pinia + Vue Router. dev 환경에서는 `/api` → `127.0.0.1:8000` 프록시.

> 본 시스템은 i.i.d. 무작위 사건인 로또 추첨을 다루므로 **이론적으로 예측 우위가 존재하지 않습니다.** `evaluate.py` 의 랜덤 베이스라인 비교는 이 한계를 정량적으로 드러내기 위한 장치입니다.

## 2. 디렉토리 구조

```
lotto-prediction/
├── backend/
│   ├── app/                    # FastAPI 애플리케이션
│   │   ├── main.py             # FastAPI factory + lifespan
│   │   ├── config.py           # Settings (호스트/포트/CORS, env-driven)
│   │   ├── deps.py             # Depends: DataStore/ModelStore
│   │   ├── schemas.py          # Pydantic 요청/응답 모델
│   │   ├── routers/            # /api/health, /draws, /models, /predict, /train
│   │   └── services/           # DataStore, ModelStore, TrainingService, prediction
│   ├── tests/                  # pytest (순수 헬퍼 단위 테스트)
│   ├── analysis.py             # 패턴 분석 + 앙상블 투표
│   ├── data_loader.py          # 엑셀 로드 + 시퀀스 전처리 + tf.data 파이프라인
│   ├── model.py                # LSTM 빌더/학습/추론
│   ├── predict.py              # CLI 예측 (번들 또는 단일 모델 입력)
│   ├── evaluate.py             # 워크포워드 백테스트 + 랜덤 베이스라인
│   ├── utils.py                # 번들 저장/로드, 시드, GPU 모니터링, 헬퍼
│   ├── visualization.py        # matplotlib 시각화
│   ├── constants.py            # 번호 범위/컬럼명 상수
│   ├── paths.py                # backend/ 기준 절대경로 상수
│   ├── env_setup.py            # TF 결정성 환경변수 (TF import 이전 호출)
│   ├── run_api.py              # uvicorn 진입 스크립트
│   ├── main.py                 # CLI 학습/예측 진입
│   ├── lotto.xlsx              # 학습 데이터
│   ├── requirements.txt
│   └── models/                 # bundle_<ts>/  + active.json (런타임 생성)
│
├── frontend/
│   ├── src/
│   │   ├── api/client.ts       # axios 인스턴스 + 백엔드 스키마 1:1 타입
│   │   ├── stores/app.ts       # Pinia: health/models/predict/training 상태
│   │   ├── router/index.ts     # /, /models, /train, /stats
│   │   ├── views/              # DashboardView · ModelsView · TrainView · StatsView
│   │   ├── components/         # NumberBalls 등
│   │   ├── App.vue · main.ts
│   │   └── assets/main.css     # Tailwind entry
│   ├── vite.config.ts · vitest.config.ts · tailwind.config.js
│   └── package.json
│
└── docs/                       # ← 본 문서 위치
```

## 3. 계층 모델 (Layered View)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       Presentation (Vue 3 SPA)                           │
│  Views(Dashboard/Models/Train/Stats) ─ Components(NumberBalls) ─ Router  │
│                              │ Pinia store (app)                         │
│                              ▼                                           │
│                       axios `apiClient` (/api/*)                         │
└──────────────────────────────────────────────────────────────────────────┘
                               │  HTTP/JSON (dev: Vite proxy)
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          API (FastAPI)                                   │
│  routers/  health · draws · models · predict · train                     │
│  schemas.py  (Pydantic; 한국어 컬럼 → 영어 키 변환 지점)                  │
│  deps.py     (DataStore / ModelStore Depends)                            │
└──────────────────────────────────────────────────────────────────────────┘
                               │  in-process 호출 (services)
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       Services (단일 인스턴스, RLock)                     │
│  DataStore   lotto.xlsx 캐시 + reload()                                  │
│  ModelStore  활성 (model, scaler, meta) + active.json 영속화             │
│  TrainingService  ThreadPoolExecutor(max=1) 직렬 학습 잡 + 진행률 콜백   │
│  prediction.make_prediction()  한 사이클 예측 오케스트레이션              │
└──────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       ML Core (CLI 와 공용)                               │
│  data_loader  load_data · preprocess_data · get_latest_sequence          │
│              create_optimized_dataset (cache/shuffle/prefetch)            │
│  model       build_model · train_and_evaluate · predict_next_numbers     │
│              export_quantized_model (TFLite)                              │
│  analysis    analyze_patterns · ensemble_prediction · analyze_number_trends│
│  utils       save/load_training_bundle · data_hash_matches               │
│              release_model_resources · set_global_seeds · setup_gpu      │
│  visualization · evaluate (워크포워드 백테스트)                            │
└──────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       Storage (로컬 파일시스템)                            │
│  backend/lotto.xlsx                  학습 원본 데이터                     │
│  backend/models/active.json          마지막 활성 번들 이름                 │
│  backend/models/bundle_<ts>/         model.(keras|h5) · scaler.pkl · meta │
│  backend/logs/tensorboard/           학습 로그                            │
│  backend/visualization/              학습 곡선/평가 차트 PNG               │
└──────────────────────────────────────────────────────────────────────────┘
```

## 4. 주요 컴포넌트

### 4.1 Backend — FastAPI 계층

| 모듈 | 역할 |
|---|---|
| `app/main.py` | `create_app()` 팩토리. `lifespan` 에서 `DataStore`/`ModelStore`/`TrainingService` 단일 인스턴스를 만들고 `active.json` 의 번들을 복원. CORS 는 `settings.cors_origins` 만 허용. |
| `app/config.py` | `Settings` (frozen dataclass). `LOTTO_API_HOST`(기본 `127.0.0.1`) / `LOTTO_API_PORT` / `LOTTO_API_CORS_ORIGINS` 환경변수로 오버라이드. |
| `app/deps.py` | `get_data_store` / `get_model_store` — `request.app.state` 에서 인스턴스 추출. |
| `app/schemas.py` | Pydantic 응답/요청 모델. 백엔드 한국어 컬럼명을 응답 경계에서 영어 키로 변환. |
| `app/routers/*` | 경량 컨트롤러. 검증/HTTP 변환만 하고 비즈니스 로직은 services 로 위임. |

### 4.2 Backend — Service 계층

| 클래스 | 책임 | 동시성 |
|---|---|---|
| `DataStore` | `lotto.xlsx` lazy load + `reload()`. | `threading.Lock` |
| `ModelStore` | 활성 `(model, scaler, meta, name)` 보관. `activate(name)` 시 ① 기존 자원 해제 → `release_model_resources()`(`clear_session()`+`gc`) → ② 새 번들 로드 → ③ `active.json` 영속화. 경로 traversal 방지를 위해 `MODELS_DIR` 직계 자식만 허용. | `threading.RLock` |
| `TrainingService` | 학습 잡 제출/실행. `ThreadPoolExecutor(max_workers=1)` 로 직렬화. `_EpochProgressCallback` 으로 에포크 진행률을 잡 상태에 반영. 완료 시 `save_training_bundle()` 호출, `auto_activate=True` 면 새 번들을 활성화. | `threading.RLock` + 단일 워커 |
| `prediction.make_prediction()` | 한 사이클 예측: 시드 세팅 → 최신 시퀀스 추출 → LSTM 예측 → 빈도 분석 → 앙상블 투표 → 보조 추천 세트 생성. | 호출자가 model snapshot 을 잡고 들어와 짧게 사용. |

### 4.3 Backend — ML 코어

- **`data_loader.preprocess_data`** — 입력 DataFrame 을 역정렬(과거→최신)한 뒤 `MinMaxScaler` 로 정규화, `sequence_length` 슬라이딩 윈도우로 `(X, y)` 생성. `create_optimized_dataset` 은 `cache → shuffle(seed) → batch → prefetch` 로 `tf.data` 파이프라인을 구성.
- **`model.build_model`** — `Bidirectional(LSTM 192) → BN → Dropout → Bidirectional(LSTM 128) → BN → Dropout → Dense(96) → BN → Dropout → Dense(6, dtype=float32)`. 출력층만 `float32` 로 고정해 `mixed_float16` 정책에서의 수치 안정성을 확보.
- **`model.train_and_evaluate`** — 마지막 20%(시계열 검증 split) 를 val 로 분리, `EarlyStopping(patience=30, restore_best_weights)` · `ReduceLROnPlateau` · `TensorBoard` + 외부 콜백 주입. 학습 곡선 PNG 저장.
- **`model.predict_next_numbers`** — 정규화된 출력을 역변환 후, 위치별 예측을 보존하되 중복 시 *가장 가까운 미사용 번호*로 결정적으로 재할당 (입력→출력 결정성 보장).
- **`analysis.ensemble_prediction`** — 4 채널 가중 투표: LSTM(0.4) · 빈도(0.2) · 최근 패턴(0.2) · 균형(0.2). 득표 상위 6 개 선정.
- **`utils`** — 학습 산출물을 `bundle_<ts>/{model.keras|h5, scaler.pkl, meta.json}` 으로 묶어 저장하고 `data_sha256` 으로 데이터 fingerprint 를 남김. `data_hash_matches` 는 UI 의 "stale 모델" 배지에 사용됨.
- **`env_setup.set_deterministic_env`** — `TF_DETERMINISTIC_OPS=1`, `TF_CUDNN_DETERMINISTIC=1` 을 **TF import 이전**에 세팅. `run_api.py` / 각 CLI 진입점이 책임.

### 4.4 Frontend (Vue 3 SPA)

| 계층 | 모듈 | 역할 |
|---|---|---|
| Routing | `router/index.ts` | `/`, `/models`, `/train`, `/stats` 4-라우트. 각 view 는 lazy import. |
| State | `stores/app.ts` (Pinia) | `health`, `models`/`activeName`, `recentDraws`, `latestPrediction`, `stats`, `trainingJob` 등 보관. 학습 잡은 `setTimeout` 폴링(`startTrainingPoll`, 1.5s 간격) — 완료 시 모델 목록 + health 재조회. |
| API | `api/client.ts` | axios `baseURL: '/api'`. `app/schemas.py` 와 1:1 매칭되는 TypeScript 인터페이스. `asApiError` 로 axios 에러 → `{ status, detail }` 정규화. |
| Views | `DashboardView` · `ModelsView` · `TrainView` · `StatsView` | 각각 예측 실행 / 번들 활성화 / 학습 트리거 + 진행률 / Chart.js 통계. |
| Component | `NumberBalls.vue` | 번호 → 색상(1-10/11-20/21-30/31-40/41-45) 매핑 표시. |

## 5. 핵심 시나리오 (Sequence Highlights)

### 5.1 부팅
1. `python run_api.py` → `set_deterministic_env()` (TF 환경변수) → `uvicorn` 기동.
2. `lifespan`: `DataStore`/`ModelStore` 인스턴스 → `ModelStore.restore_from_disk()` 가 `models/active.json` 을 읽어 마지막 번들을 활성화 (실패해도 앱은 기동).
3. SPA(`App.vue` mount) → `fetchHealth()` → `/api/health` (TF 버전 + GPU 가용성 표시).

### 5.2 예측 (`POST /api/predict`)
1. 라우터에서 `TrainingService.is_running()` 확인 — `True` 면 `409` (GPU 자원 충돌 방지).
2. `ModelStore.snapshot()` 으로 `(model, scaler, meta, name)` 획득 — 없으면 `400`.
3. `sequence_length` 결정(요청 우선, 없으면 `meta`).
4. `prediction.make_prediction()`:
   - `set_global_seeds(seed)` → `get_latest_sequence(df, …, scaler)`
   - `predict_next_numbers(model, …)` (LSTM)
   - `analyze_patterns(df)` → `ensemble_prediction(df, lstm, …)`
   - `suggest_balanced_numbers(...)` 로 보조 추천 세트
5. 응답에 `data_hash_match`(번들 학습 시점 데이터 vs 현재 xlsx SHA-256 비교) 포함.

### 5.3 학습 (`POST /api/train`)
1. 잡 제출(`is_running()=True` 면 `409`) → `ThreadPoolExecutor` 에 `_run` 디스패치.
2. 워커 스레드:
   - `set_global_seeds` → `preprocess_data` → `train_and_evaluate(...)` (콜백 `_EpochProgressCallback` 가 매 에포크에 `epoch / loss / val_loss / best_val_loss` 업데이트).
   - `save_training_bundle()` 으로 `models/bundle_<ts>/` 디렉토리 생성.
   - `auto_activate=True` 면 `ModelStore.activate(bundle_name)` 호출 — 기존 모델 자원 해제 후 새 번들 로드.
3. 프런트는 `/api/train` 폴링으로 진행률 표시, `running/queued` 가 아닌 상태가 되면 모델 목록 갱신.

### 5.4 모델 활성화 (`POST /api/models/active`)
1. `name` 디렉토리가 `MODELS_DIR` 직계 자식인지 검증 (`..` traversal 차단).
2. 기존 model/scaler/meta 참조 → `None`, `release_model_resources()` (Keras session clear + `gc.collect`).
3. 새 번들 로드 → `active.json` 갱신.
4. 응답에 `data_hash_match` 포함하여 UI 가 "데이터 변경됨" 배지 노출.

## 6. 도메인/데이터 모델

- **번호 범위**: `MIN_NUMBER=1`, `MAX_NUMBER=45`, `NUM_BALLS=6` (`constants.py`).
- **컬럼명**: `회차`, `번호1`…`번호6` (xlsx 원본). 응답 경계에서 `draw_no`, `numbers[]` 로 변환.
- **데이터 정렬 규약**: `load_data` 는 내림차순(최신→과거). 학습 시 `preprocess_data` 가 시계열 학습을 위해 다시 뒤집어 오름차순으로 변환.

### 학습 번들 (`bundle_<ts>/`)
```
bundle_<YYYYmmdd_HHMMSS>/
├── model.keras            # TF<2.13 이면 model.h5 로 폴백
├── scaler.pkl             # MinMaxScaler (sklearn pickle)
└── meta.json              # {
                            #   schema_version, timestamp, model_file, scaler_file,
                            #   data_file, data_sha256,
                            #   sequence_length, seed,
                            #   tensorflow_version, sklearn_version, numpy_version,
                            #   epochs?, batch_size?, best_val_loss?, trained_via?
                            # }
```

## 7. 동시성/리소스 정책

- **단일 사용자 MVP** 가정. 인증/멀티테넌시 없음. 기본 바인딩 `127.0.0.1`.
- **학습 ↔ 예측 상호배제** — 학습 잡 진행 중 `/api/predict` 는 `409`. GPU 메모리 충돌 회피.
- **학습 직렬화** — `ThreadPoolExecutor(max_workers=1)`. 동시에 두 학습 제출 시 두 번째도 `409`.
- **모델 교체 시 자원 해제** — 참조 떨구기 → `tf.keras.backend.clear_session()` → `gc.collect()`. 이 순서를 어기면 GPU 메모리가 회수되지 않을 수 있음.
- **결정성** — `TF_DETERMINISTIC_OPS=1` 환경변수 + 모든 진입점에서 `set_global_seeds(seed)` (`random` / `numpy` / `tf`). `tf.data.shuffle(seed=...)` 명시. 완전한 결정성에는 셸에서 `PYTHONHASHSEED=0` 필요.

## 8. 보안/안전 고려사항

| 영역 | 조치 |
|---|---|
| 네트워크 노출 | 기본 호스트 `127.0.0.1`. LAN 노출은 `LOTTO_API_HOST=0.0.0.0` 환경변수로만 명시적 허용. |
| CORS | `LOTTO_API_CORS_ORIGINS` 화이트리스트 (dev 기본: `localhost:5173`, `127.0.0.1:5173`). `allow_credentials=False`. |
| 경로 traversal | `ModelStore.activate` 가 `MODELS_DIR` 직계 자식인지 검증. |
| 데이터 무결성 | 번들 메타의 `data_sha256` 과 현재 `lotto.xlsx` SHA-256 비교 → 응답 `data_hash_match` 플래그. |
| 자원 누수 | 모델 교체 시 `release_model_resources()`. `matplotlib` 은 `Agg` 백엔드 + `plt.close(fig)`. |

## 9. 테스트 전략

- **Backend (pytest)** — `tests/test_pure_helpers.py` (18 케이스 가량): `validate_lotto_numbers`, `compare_predictions`, `_sha256_file`, `data_hash_matches`, `suggest_balanced_numbers`, `_random_baseline_matches`. `tests/test_evaluate_baseline.py` 는 평가 베이스라인을 검증. TF import 비용으로 첫 컬렉션이 다소 느림.
- **Frontend (Vitest)** — `NumberBalls` 색상 매핑, `asApiError` 파싱 등 단위 테스트.
- **수동/탐색** — UI 에서 학습 → 활성화 → 예측 시나리오 / 데이터 교체 후 `data_hash_match=false` 배지 확인.

## 10. 확장 시 주의점 (Known Trade-offs)

- **잡 상태 휘발성** — `TrainingService` 의 잡 딕셔너리는 인메모리. 프로세스 재기동 시 사라짐. 영속성이 필요해지면 SQLite/JSON 파일 백킹으로 교체.
- **단일 활성 모델** — `ModelStore` 는 한 번에 한 번들만 활성. A/B 비교는 별도 엔드포인트로 확장 필요.
- **번들 정리 없음** — `models/` 가 무한 누적됨. 추후 retention/GC 정책 필요.
- **데이터 업로드 API 부재** — `lotto.xlsx` 수동 교체 후 `DataStore.reload()` 트리거 라우터는 미구현. V3 후보.
- **예측의 본질적 한계** — `evaluate.py` 의 랜덤 베이스라인 비교 결과를 정기적으로 노출해야 사용자가 모델의 "예측"을 과신하지 않음.
