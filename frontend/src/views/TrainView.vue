<script setup lang="ts">
import { computed, onMounted, onUnmounted, reactive } from 'vue'

import type { TrainRequest } from '../api/client'
import { useAppStore } from '../stores/app'

const store = useAppStore()

const form = reactive<TrainRequest>({
  epochs: 100,
  batch_size: 64,
  sequence_length: 10,
  seed: 42,
  auto_activate: true,
})

const progressPct = computed(() => {
  const j = store.trainingJob
  if (!j || !j.total_epochs) return 0
  return Math.min(100, Math.round((j.epoch / j.total_epochs) * 100))
})

const elapsedLabel = computed(() => {
  const j = store.trainingJob
  if (!j?.started_at) return null
  const end = j.finished_at ?? Date.now() / 1000
  const s = Math.max(0, end - j.started_at)
  const m = Math.floor(s / 60)
  return `${m}m ${Math.floor(s % 60)}s`
})

const submitting = computed(
  () => store.trainingJob?.status === 'queued' || store.trainingJob?.status === 'running'
)

function submit() {
  store.submitTraining({ ...form })
}

onMounted(() => {
  store.refreshTrainingJob().then(() => {
    const j = store.trainingJob
    if (j && (j.status === 'queued' || j.status === 'running')) store.startTrainingPoll()
  })
})

onUnmounted(() => store.stopTrainingPoll())
</script>

<template>
  <div class="space-y-4">
    <section class="bg-white border border-slate-200 rounded p-4">
      <h2 class="font-medium text-base mb-3">새 모델 학습</h2>

      <div class="grid grid-cols-2 md:grid-cols-5 gap-3 text-sm">
        <label class="flex flex-col">
          <span class="text-xs text-slate-500 mb-1">에포크</span>
          <input v-model.number="form.epochs" type="number" min="1" max="2000"
                 class="border border-slate-300 rounded px-2 py-1" />
        </label>
        <label class="flex flex-col">
          <span class="text-xs text-slate-500 mb-1">배치 크기</span>
          <input v-model.number="form.batch_size" type="number" min="1" max="1024"
                 class="border border-slate-300 rounded px-2 py-1" />
        </label>
        <label class="flex flex-col">
          <span class="text-xs text-slate-500 mb-1">시퀀스 길이</span>
          <input v-model.number="form.sequence_length" type="number" min="2" max="50"
                 class="border border-slate-300 rounded px-2 py-1" />
        </label>
        <label class="flex flex-col">
          <span class="text-xs text-slate-500 mb-1">시드</span>
          <input v-model.number="form.seed" type="number"
                 class="border border-slate-300 rounded px-2 py-1" />
        </label>
        <label class="flex items-end gap-2">
          <input v-model="form.auto_activate" type="checkbox" class="rounded" />
          <span class="text-xs">완료 시 자동 활성화</span>
        </label>
      </div>

      <div class="mt-3 flex items-center gap-3">
        <button @click="submit" :disabled="submitting"
                class="px-3 py-1.5 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:bg-slate-400">
          {{ submitting ? '진행 중…' : '학습 시작' }}
        </button>
        <span v-if="store.trainingError" class="text-sm text-red-600">{{ store.trainingError }}</span>
      </div>

      <p class="mt-3 text-xs text-slate-500">
        학습이 진행되는 동안 <code class="bg-slate-100 px-1 rounded">/api/predict</code>
        는 409 로 거절됩니다 (GPU 메모리 직렬화). EarlyStopping 으로 max epoch 이전에 끝날 수 있습니다.
      </p>
    </section>

    <section v-if="store.trainingJob" class="bg-white border border-slate-200 rounded p-4">
      <div class="flex items-center justify-between mb-2">
        <h2 class="font-medium text-base">잡 상태</h2>
        <span :class="{
          'text-blue-600': store.trainingJob.status === 'running',
          'text-slate-500': store.trainingJob.status === 'queued',
          'text-emerald-600': store.trainingJob.status === 'completed',
          'text-red-600': store.trainingJob.status === 'failed',
        }" class="text-xs font-medium uppercase">
          {{ store.trainingJob.status }}
        </span>
      </div>

      <div class="text-xs text-slate-500 mb-3">
        <code>{{ store.trainingJob.job_id }}</code>
        <span v-if="elapsedLabel"> · 경과 {{ elapsedLabel }}</span>
      </div>

      <div class="mb-2 text-sm">
        에포크 <span class="font-medium">{{ store.trainingJob.epoch }}</span>
        / {{ store.trainingJob.total_epochs }}
      </div>
      <div class="w-full bg-slate-100 rounded h-2 overflow-hidden">
        <div class="bg-blue-500 h-full transition-all" :style="{ width: `${progressPct}%` }"></div>
      </div>

      <div class="mt-3 grid grid-cols-2 md:grid-cols-3 gap-3 text-xs text-slate-600">
        <div>
          <div class="text-slate-400">last loss</div>
          <div>{{ store.trainingJob.last_loss?.toFixed?.(6) ?? '—' }}</div>
        </div>
        <div>
          <div class="text-slate-400">last val_loss</div>
          <div>{{ store.trainingJob.last_val_loss?.toFixed?.(6) ?? '—' }}</div>
        </div>
        <div>
          <div class="text-slate-400">best val_loss</div>
          <div class="text-emerald-700">{{ store.trainingJob.best_val_loss?.toFixed?.(6) ?? '—' }}</div>
        </div>
      </div>

      <div v-if="store.trainingJob.status === 'completed' && store.trainingJob.bundle_name"
           class="mt-4 text-sm text-emerald-800 bg-emerald-50 border border-emerald-200 rounded px-3 py-2">
        새 번들 <code class="bg-white px-1.5 rounded">{{ store.trainingJob.bundle_name }}</code> 가 생성되었습니다.
        <span v-if="store.trainingJob.params.auto_activate"> 자동 활성화됨.</span>
      </div>

      <div v-if="store.trainingJob.status === 'failed'"
           class="mt-4 text-sm text-red-800 bg-red-50 border border-red-200 rounded px-3 py-2">
        실패: {{ store.trainingJob.error }}
      </div>
    </section>
  </div>
</template>
