import { defineStore } from 'pinia'

import {
  apiClient,
  asApiError,
  type BundleSummary,
  type DrawRow,
  type HealthResponse,
  type PredictResponse,
  type StatsResponse,
  type TrainJob,
  type TrainRequest,
} from '../api/client'

interface State {
  health: HealthResponse | null
  models: BundleSummary[]
  activeName: string | null
  recentDraws: DrawRow[]
  totalDraws: number
  latestPrediction: PredictResponse | null
  isPredicting: boolean
  error: string | null
  stats: StatsResponse | null
  trainingJob: TrainJob | null
  trainingPollTimer: number | null
  trainingError: string | null
}

export const useAppStore = defineStore('app', {
  state: (): State => ({
    health: null,
    models: [],
    activeName: null,
    recentDraws: [],
    totalDraws: 0,
    latestPrediction: null,
    isPredicting: false,
    error: null,
    stats: null,
    trainingJob: null,
    trainingPollTimer: null,
    trainingError: null,
  }),

  actions: {
    async fetchHealth() {
      try {
        this.health = await apiClient.health()
      } catch {
        // health 는 best-effort: 실패해도 UI 는 계속.
      }
    },

    async fetchRecentDraws(limit = 20) {
      const r = await apiClient.recentDraws(limit)
      this.recentDraws = r.rows
      this.totalDraws = r.total_draws
    },

    async fetchModels() {
      const r = await apiClient.listModels()
      this.models = r.bundles
      this.activeName = r.active_name
    },

    async activateModel(name: string) {
      const r = await apiClient.activateModel(name)
      this.activeName = r.active_name
      await this.fetchModels()
      return r
    },

    async runPrediction(seed = 42, numSets = 3) {
      this.isPredicting = true
      this.error = null
      try {
        this.latestPrediction = await apiClient.predict({ seed, num_sets: numSets })
      } catch (e) {
        this.error = asApiError(e).detail
      } finally {
        this.isPredicting = false
      }
    },

    async fetchStats() {
      this.stats = await apiClient.stats()
    },

    async submitTraining(body: TrainRequest) {
      this.trainingError = null
      try {
        const job = await apiClient.submitTraining(body)
        this.trainingJob = job
        this.startTrainingPoll()
      } catch (e) {
        this.trainingError = asApiError(e).detail
      }
    },

    async refreshTrainingJob() {
      const list = await apiClient.listTraining()
      if (list.current_job_id) {
        this.trainingJob = list.jobs.find((j) => j.job_id === list.current_job_id) ?? this.trainingJob
      } else if (list.jobs.length > 0) {
        // 가장 최근 잡 표시 (완료/실패 포함)
        this.trainingJob = list.jobs[0]
      }
    },

    startTrainingPoll(intervalMs = 1500) {
      this.stopTrainingPoll()
      const tick = async () => {
        try {
          await this.refreshTrainingJob()
        } catch {
          // 일시적 네트워크 에러는 무시하고 다음 tick 으로
        }
        const j = this.trainingJob
        if (j && (j.status === 'running' || j.status === 'queued')) {
          this.trainingPollTimer = window.setTimeout(tick, intervalMs)
        } else {
          // 완료/실패: 모델 목록 + 활성도 새로고침해 UI 동기화
          this.trainingPollTimer = null
          await Promise.all([this.fetchModels(), this.fetchHealth()])
        }
      }
      tick()
    },

    stopTrainingPoll() {
      if (this.trainingPollTimer !== null) {
        window.clearTimeout(this.trainingPollTimer)
        this.trainingPollTimer = null
      }
    },
  },
})
