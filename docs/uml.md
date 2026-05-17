# UML 다이어그램

> 모든 다이어그램은 [Mermaid](https://mermaid.js.org/) 문법으로 작성되었습니다. GitHub/GitLab/대부분의 마크다운 뷰어에서 바로 렌더링됩니다.

## 1. 컨텍스트 / 컴포넌트 다이어그램 (C4-Light)

```mermaid
graph LR
  User[👤 사용자]
  Browser[🌐 브라우저<br/>Vue 3 SPA]
  API[(⚙ FastAPI<br/>uvicorn @ 127.0.0.1:8000)]
  ML[🧠 ML Core<br/>analysis · model · data_loader · utils]
  FS[(💾 backend/<br/>lotto.xlsx<br/>models/bundle_*/<br/>active.json)]
  GPU[🖥 NVIDIA GPU<br/>TF + CUDA + cuDNN]
  TB[(📈 TensorBoard logs)]

  User --> Browser
  Browser -->|/api/* HTTP-JSON| API
  API --> ML
  ML -->|학습/추론| GPU
  ML --> FS
  ML --> TB

  classDef ext fill:#eef,stroke:#88c
  classDef store fill:#fef9e0,stroke:#caa
  class User,GPU ext
  class FS,TB store
```

## 2. 백엔드 컴포넌트 구조

```mermaid
graph TB
  subgraph Routers["app/routers/"]
    H[health.py]
    D[draws.py]
    M[models.py]
    P[predict.py]
    T[training.py]
  end

  subgraph Services["app/services/"]
    DS[DataStore]
    MS[ModelStore]
    TS[TrainingService]
    PR[prediction.make_prediction]
  end

  subgraph Schemas["app/"]
    SC[schemas.py<br/>Pydantic models]
    CF[config.py · deps.py]
  end

  subgraph Core["ML Core (backend/)"]
    DL[data_loader]
    MD[model]
    AN[analysis]
    UT[utils]
    EV[env_setup]
    PT[paths · constants]
  end

  H --> SC
  D --> DS
  D --> AN
  M --> MS
  M --> DS
  P --> MS
  P --> DS
  P --> TS
  P --> PR
  T --> TS

  DS --> DL
  MS --> UT
  TS --> DS
  TS --> MS
  TS --> DL
  TS --> MD
  TS --> UT

  PR --> DL
  PR --> MD
  PR --> AN
  PR --> UT

  MD --> DL
  MD --> PT
  AN --> PT
  UT --> PT
  EV -.보장.-> MD
```

## 3. 클래스 다이어그램 (Backend Service Layer)

```mermaid
classDiagram
  direction LR

  class Settings {
    +str host = "127.0.0.1"
    +int port = 8000
    +list~str~ cors_origins
    +int default_sequence_length = 10
    +int default_seed = 42
    +int default_recent_limit = 20
    +int max_recent_limit = 200
  }

  class DataStore {
    -Path _data_path
    -DataFrame _df
    -Lock _lock
    +data_path: Path
    +get() DataFrame
    +reload() DataFrame
  }

  class ModelStore {
    -RLock _lock
    -Any _model
    -Any _scaler
    -dict _meta
    -str _active_name
    +active_name: str
    +snapshot() tuple
    +list_bundles(data_path) list
    +activate(name) tuple
    +restore_from_disk() str
    -_persist() void
  }

  class TrainingParams {
    +int epochs = 300
    +int batch_size = 64
    +int sequence_length = 10
    +int seed = 42
    +bool auto_activate = True
  }

  class TrainingJob {
    +str job_id
    +TrainingParams params
    +str status
    +int epoch
    +int total_epochs
    +float best_val_loss
    +float last_loss
    +float last_val_loss
    +str error
    +str bundle_name
    +float submitted_at
    +float started_at
    +float finished_at
    +to_dict() dict
  }

  class TrainingService {
    -DataStore _data_store
    -ModelStore _model_store
    -dict _jobs
    -RLock _lock
    -ThreadPoolExecutor _executor
    -str _current_job_id
    +get(job_id) TrainingJob
    +list() list~TrainingJob~
    +is_running() bool
    +current_job() TrainingJob
    +submit(params) TrainingJob
    -_run(job) void
  }

  class _EpochProgressCallback {
    -TrainingJob _job
    +on_epoch_end(epoch, logs) void
  }

  class PredictionResult {
    +list~int~ lstm
    +list~int~ ensemble
    +list~list~int~~ additional_sets
    +int sequence_length
    +int seed
  }

  class make_prediction {
    <<function>>
    +(model, scaler, df, sequence_length, seed, num_sets) PredictionResult
  }

  TrainingService --> DataStore : uses
  TrainingService --> ModelStore : uses
  TrainingService "1" o-- "*" TrainingJob : tracks
  TrainingJob --> TrainingParams
  TrainingService ..> _EpochProgressCallback : creates per job
  _EpochProgressCallback --> TrainingJob : updates progress
  make_prediction ..> PredictionResult : returns
  ModelStore ..> Settings : reads (indirect via paths)
```

## 4. 클래스 다이어그램 (ML 코어 핵심 함수)

```mermaid
classDiagram
  direction TB

  class data_loader {
    <<module>>
    +load_data(file_path) DataFrame
    +preprocess_data(df, seq_len, use_tf_dataset) tuple
    +create_optimized_dataset(X, y, batch_size, shuffle_buffer, seed) Dataset
    +get_latest_sequence(df, seq_len, scaler) ndarray
  }

  class model {
    <<module>>
    +build_model(input_shape, use_gpu) Sequential
    +train_and_evaluate(X, y, epochs, batch_size, validation_split, use_gpu, extra_callbacks) tuple
    +predict_next_numbers(model, latest_sequence, scaler) list~int~
    +export_quantized_model(model, output_path) str
  }

  class analysis {
    <<module>>
    +analyze_patterns(df) tuple
    +ensemble_prediction(df, lstm_prediction, top_frequencies, seq_len) list~int~
    +generate_balanced_prediction() list~int~
    +analyze_number_trends(df, window_size) tuple
  }

  class utils {
    <<module>>
    +set_global_seeds(seed) void
    +save_model(model, file_path) bool
    +load_model(file_path) Sequential
    +save_scaler(scaler, file_path) bool
    +load_scaler(file_path) MinMaxScaler
    +save_training_bundle(model, scaler, data_path, seq_len, seed, extra_meta) Path
    +load_training_bundle(bundle_dir) tuple
    +data_hash_matches(meta, data_path) bool
    +list_training_bundles(models_root) list~Path~
    +release_model_resources() void
    +setup_gpu() tuple
    +setup_gpu_monitoring(interval) tuple
    +compare_predictions(actual, predicted) tuple
    +validate_lotto_numbers(numbers) tuple
    +suggest_balanced_numbers(frequencies, recent, n) list
  }

  class env_setup {
    <<module>>
    +set_deterministic_env() void
  }

  class paths {
    <<module, constants>>
    +BASE_DIR: Path
    +MODELS_DIR: Path
    +LOGS_DIR: Path
    +VISUALIZATION_DIR: Path
    +TENSORBOARD_DIR: Path
    +DEFAULT_DATA_FILE: Path
  }

  class constants {
    <<module, constants>>
    +NUM_BALLS = 6
    +MIN_NUMBER = 1
    +MAX_NUMBER = 45
    +LOTTO_COLUMNS: list
    +DRAW_COLUMN = "회차"
  }

  model --> data_loader
  model --> paths
  model --> constants
  analysis --> constants
  utils --> paths
  utils --> constants
  data_loader --> constants
```

## 5. 시퀀스 — 예측 (`POST /api/predict`)

```mermaid
sequenceDiagram
  autonumber
  participant FE as Vue Store
  participant R as predict router
  participant TS as TrainingService
  participant MS as ModelStore
  participant DS as DataStore
  participant PR as make_prediction()
  participant ML as ML Core<br/>(data_loader · model · analysis)

  FE->>R: POST /api/predict { seed, num_sets }
  R->>TS: is_running()
  alt 학습 중
    TS-->>R: True
    R-->>FE: 409 학습 중
  else
    TS-->>R: False
    R->>MS: snapshot()
    MS-->>R: (model, scaler, meta, name)
    alt 활성 모델 없음
      R-->>FE: 400 활성 모델 없음
    else
      R->>DS: get() (DataFrame)
      DS-->>R: df
      R->>PR: make_prediction(model, scaler, df, seq, seed, num_sets)
      PR->>ML: set_global_seeds(seed)
      PR->>ML: get_latest_sequence(df, seq, scaler)
      ML-->>PR: latest_sequence
      PR->>ML: predict_next_numbers(model, latest_sequence, scaler)
      ML-->>PR: lstm[6]
      PR->>ML: analyze_patterns(df)
      ML-->>PR: (frequencies, odd_even, range_patterns)
      PR->>ML: ensemble_prediction(df, lstm, frequencies, seq)
      ML-->>PR: ensemble[6]
      PR->>ML: suggest_balanced_numbers(...)
      ML-->>PR: additional_sets[][]
      PR-->>R: PredictionResult
      R->>ML: data_hash_matches(meta, data_path)
      ML-->>R: bool
      R-->>FE: 200 PredictResponse
    end
  end
```

## 6. 시퀀스 — 학습 (`POST /api/train`)

```mermaid
sequenceDiagram
  autonumber
  participant FE as Vue Store
  participant R as train router
  participant TS as TrainingService
  participant EX as ThreadPoolExecutor<br/>(max_workers=1)
  participant J as TrainingJob
  participant DS as DataStore
  participant MD as model.train_and_evaluate
  participant CB as _EpochProgressCallback
  participant UT as utils.save_training_bundle
  participant MS as ModelStore

  FE->>R: POST /api/train { epochs, batch, seq, seed, auto_activate }
  R->>TS: is_running()
  alt 진행 중
    TS-->>R: True
    R-->>FE: 409
  else
    TS-->>R: False
    R->>TS: submit(params)
    TS->>J: new TrainingJob(queued)
    TS->>EX: submit(_run, job)
    TS-->>R: TrainingJob
    R-->>FE: 200 TrainJob

    EX->>J: status=running
    EX->>DS: get() → df
    EX->>EX: preprocess_data(df, seq) → (X, y, scaler)
    EX->>MD: train_and_evaluate(X, y, epochs, batch, extra_callbacks=[CB])
    loop 각 epoch
      MD->>CB: on_epoch_end(epoch, logs)
      CB->>J: update epoch/loss/val_loss/best_val_loss
    end
    MD-->>EX: (model, history)
    EX->>UT: save_training_bundle(model, scaler, ...)
    UT-->>EX: bundle_dir
    EX->>J: status=completed, bundle_name=...
    opt auto_activate
      EX->>MS: activate(bundle_dir.name)
      MS-->>EX: ok
    end

    par FE 폴링
      FE->>R: GET /api/train (every 1.5s)
      R->>TS: list() / current_job()
      TS-->>R: jobs
      R-->>FE: TrainJobListResponse
    end
  end
```

## 7. 시퀀스 — 모델 활성화 (`POST /api/models/active`)

```mermaid
sequenceDiagram
  autonumber
  participant FE as Vue Store
  participant R as models router
  participant MS as ModelStore
  participant FS as filesystem<br/>(models/bundle_*/)
  participant TF as Keras/TF
  participant DS as DataStore
  participant UT as utils

  FE->>R: POST /api/models/active { name }
  R->>MS: activate(name)
  MS->>MS: resolve(MODELS_DIR/name)<br/>parent == MODELS_DIR ?
  alt traversal/없음
    MS-->>R: FileNotFoundError
    R-->>FE: 404
  else
    MS->>MS: drop refs (_model=_scaler=_meta=None)
    MS->>TF: tf.keras.backend.clear_session()
    MS->>TF: gc.collect()
    MS->>FS: load model.(keras|h5) + scaler.pkl + meta.json
    FS-->>MS: (model, scaler, meta)
    MS->>MS: _active_name = name
    MS->>FS: write active.json
    MS-->>R: (meta, hash_match, bundle_dir)
    R->>DS: data_path
    R->>UT: data_hash_matches(meta, data_path)
    UT-->>R: bool
    R-->>FE: 200 ActivateBundleResponse
  end
```

## 8. 시퀀스 — 부팅 (lifespan)

```mermaid
sequenceDiagram
  autonumber
  participant SH as shell
  participant EP as run_api.py
  participant ENV as env_setup
  participant UV as uvicorn
  participant APP as FastAPI
  participant DS as DataStore
  participant MS as ModelStore
  participant TS as TrainingService

  SH->>EP: python run_api.py
  EP->>ENV: set_deterministic_env()
  Note over ENV: TF_DETERMINISTIC_OPS=1<br/>TF_CUDNN_DETERMINISTIC=1
  EP->>UV: uvicorn.run("app.main:app", host, port)
  UV->>APP: import + create_app()
  APP->>APP: lifespan startup
  APP->>DS: new DataStore()
  APP->>MS: new ModelStore()
  APP->>MS: restore_from_disk()
  MS->>MS: read models/active.json
  alt 활성 번들 있음
    MS->>MS: activate(name)
    MS-->>APP: name
  else 없음/실패
    MS-->>APP: None (경고 로그만)
  end
  APP->>TS: new TrainingService(DS, MS)
  APP->>APP: state.{data_store, model_store, training_service} 보관
  APP-->>UV: ready
```

## 9. 상태 다이어그램 — TrainingJob

```mermaid
stateDiagram-v2
  [*] --> queued : submit(params)
  queued --> running : executor picks up
  running --> running : on_epoch_end<br/>(epoch++, loss, val_loss)
  running --> completed : 학습 정상 종료 + bundle 저장
  running --> failed : Exception (데이터/학습 오류)
  completed --> [*]
  failed --> [*]
  completed --> [*] : (auto_activate=True 시<br/>ModelStore.activate 부수효과)
```

## 10. 상태 다이어그램 — ModelStore

```mermaid
stateDiagram-v2
  [*] --> Empty : __init__ (active_name=None)
  Empty --> Loading : activate(name)
  Loading --> Active : load_training_bundle 성공 + active.json 저장
  Loading --> Empty : FileNotFoundError / 로드 실패
  Active --> Releasing : activate(other_name)
  Releasing --> Loading : refs=None<br/>clear_session + gc.collect
  Active --> Empty : (활성 해제 API는 미구현 — 향후 확장)
  note right of Loading
    경로 traversal 차단:
    bundle_dir.parent ==
    MODELS_DIR.resolve()
  end note
```

## 11. 데이터 흐름 — 학습 데이터 파이프라인

```mermaid
flowchart LR
  X[("lotto.xlsx")] --> L["load_data<br/>내림차순 정렬"]
  L --> P[preprocess_data]
  P -->|"역정렬: 과거→최신"| S["MinMaxScaler.fit_transform"]
  S --> W["슬라이딩 윈도우<br/>seq_len"]
  W --> XY[("(X, y) float32")]
  XY --> T["create_optimized_dataset<br/>cache → shuffle(seed) → batch → prefetch"]
  T --> F["model.fit"]
  F --> B[("bundle_&lt;ts&gt;/<br/>model · scaler · meta")]
  B -.SHA-256.-> X
  B --> A[("active.json")]
```

## 12. 프런트엔드 컴포넌트/상태

```mermaid
graph LR
  subgraph Vue["Vue 3 SPA"]
    App[App.vue<br/>nav + RouterView]
    R[router/index.ts]
    subgraph Views
      DV[DashboardView]
      MV[ModelsView]
      TV[TrainView]
      SV[StatsView]
    end
    NB[NumberBalls.vue]
    Store[(Pinia useAppStore)]
    Cli[api/client.ts<br/>axios + types]
  end

  App --> R
  R --> DV
  R --> MV
  R --> TV
  R --> SV
  DV --> NB
  MV --> NB
  DV --> Store
  MV --> Store
  TV --> Store
  SV --> Store
  Store --> Cli
  Cli -->|/api/*| API[(FastAPI)]
```

## 13. 폴링 — 학습 진행률

```mermaid
sequenceDiagram
  autonumber
  participant V as TrainView
  participant S as Pinia Store
  participant API as /api/train
  V->>S: submitTraining(body)
  S->>API: POST /api/train
  API-->>S: TrainJob (queued)
  S->>S: startTrainingPoll(1500ms)
  loop status ∈ {queued, running}
    S->>API: GET /api/train
    API-->>S: TrainJobListResponse
    S->>S: trainingJob = current/latest
    S->>V: reactive update
  end
  Note over S: 완료/실패 → 폴링 중단
  S->>API: GET /api/models (refresh)
  S->>API: GET /api/health
```
