// 백엔드 ``app/schemas.py`` 와 1:1 매칭되는 타입 + axios 인스턴스.
// dev 에서는 Vite proxy 가 /api → http://127.0.0.1:8000 으로 라우팅한다.
// 프로덕션 빌드를 FastAPI 가 직접 서빙하면 baseURL 그대로 작동.

import axios, { type AxiosInstance } from 'axios'

export interface HealthResponse {
  status: string
  tensorflow_version: string | null
  gpu_available: boolean
}

export interface DrawRow {
  draw_no: number
  numbers: number[]
}

export interface RecentDrawsResponse {
  total_draws: number
  rows: DrawRow[]
}

export interface FrequencyEntry {
  number: number
  count: number
}

export interface StatsResponse {
  total_draws: number
  frequencies: FrequencyEntry[]
  odd_even: Record<string, number>
  range_distribution: Record<string, number>
}

export interface BundleSummary {
  name: string
  timestamp: string | null
  sequence_length: number | null
  seed: number | null
  data_sha256: string | null
  tensorflow_version: string | null
  sklearn_version: string | null
  is_active: boolean
  data_hash_match: boolean
}

export interface ModelsResponse {
  bundles: BundleSummary[]
  active_name: string | null
}

export interface ActivateBundleRequest {
  name: string
}

export interface ActivateBundleResponse {
  active_name: string
  data_hash_match: boolean
  message: string
}

export interface PredictRequest {
  sequence_length?: number | null
  seed: number
  num_sets: number
}

export interface PredictResponse {
  lstm: number[]
  ensemble: number[]
  additional_sets: number[][]
  active_bundle: string
  data_hash_match: boolean
  sequence_length: number
  seed: number
}

export interface TrainRequest {
  epochs: number
  batch_size: number
  sequence_length: number
  seed: number
  auto_activate: boolean
}

export interface TrainJobParams {
  epochs: number
  batch_size: number
  sequence_length: number
  seed: number
  auto_activate: boolean
}

export interface TrainJob {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed' | string
  epoch: number
  total_epochs: number
  best_val_loss: number | null
  last_loss: number | null
  last_val_loss: number | null
  error: string | null
  bundle_name: string | null
  submitted_at: number
  started_at: number | null
  finished_at: number | null
  params: TrainJobParams
}

export interface TrainJobListResponse {
  current_job_id: string | null
  jobs: TrainJob[]
}

const http: AxiosInstance = axios.create({
  baseURL: '/api',
  timeout: 60_000,
})

export const apiClient = {
  health: () => http.get<HealthResponse>('/health').then((r) => r.data),
  recentDraws: (limit = 20) =>
    http.get<RecentDrawsResponse>('/draws/recent', { params: { limit } }).then((r) => r.data),
  stats: () => http.get<StatsResponse>('/draws/stats').then((r) => r.data),
  listModels: () => http.get<ModelsResponse>('/models').then((r) => r.data),
  activateModel: (name: string) =>
    http.post<ActivateBundleResponse>('/models/active', { name }).then((r) => r.data),
  predict: (body: PredictRequest) =>
    http.post<PredictResponse>('/predict', body).then((r) => r.data),
  submitTraining: (body: TrainRequest) =>
    http.post<TrainJob>('/train', body).then((r) => r.data),
  listTraining: () => http.get<TrainJobListResponse>('/train').then((r) => r.data),
  getTrainingJob: (id: string) => http.get<TrainJob>(`/train/${id}`).then((r) => r.data),
}

export type ApiError = {
  status: number
  detail: string
}

export function asApiError(err: unknown): ApiError {
  if (axios.isAxiosError(err)) {
    const status = err.response?.status ?? 0
    const detail =
      (err.response?.data as { detail?: string } | undefined)?.detail ??
      err.message ??
      '요청 실패'
    return { status, detail }
  }
  return { status: 0, detail: (err as Error)?.message ?? '알 수 없는 오류' }
}
