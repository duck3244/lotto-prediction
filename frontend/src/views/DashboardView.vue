<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'

import NumberBalls from '../components/NumberBalls.vue'
import { useAppStore } from '../stores/app'

const store = useAppStore()
const seed = ref(42)
const numSets = ref(3)

onMounted(async () => {
  await Promise.all([store.fetchModels(), store.fetchRecentDraws(10)])
})

function runPrediction() {
  store.runPrediction(seed.value, numSets.value)
}
</script>

<template>
  <div class="space-y-6">
    <!-- Prediction card -->
    <section class="bg-white border border-slate-200 rounded p-4">
      <h2 class="font-medium text-base mb-3">다음 회차 예측</h2>

      <div v-if="!store.activeName" class="text-sm text-slate-600">
        활성 모델이 없습니다.
        <RouterLink to="/models" class="text-blue-600 underline">모델 페이지</RouterLink>
        에서 학습 번들을 선택하세요.
      </div>

      <template v-else>
        <div class="flex flex-wrap items-center gap-4 mb-3 text-sm">
          <span class="text-slate-500">활성 모델</span>
          <code class="text-xs bg-slate-100 px-2 py-1 rounded">{{ store.activeName }}</code>

          <label class="flex items-center gap-1.5">
            시드
            <input
              v-model.number="seed"
              type="number"
              class="border border-slate-300 rounded px-2 py-1 w-20 text-right"
            />
          </label>

          <label class="flex items-center gap-1.5">
            추가 세트
            <input
              v-model.number="numSets"
              type="number"
              min="0"
              max="10"
              class="border border-slate-300 rounded px-2 py-1 w-16 text-right"
            />
          </label>

          <button
            @click="runPrediction"
            :disabled="store.isPredicting"
            class="ml-auto px-3 py-1.5 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:bg-slate-400"
          >
            {{ store.isPredicting ? '예측 중…' : '예측 실행' }}
          </button>
        </div>

        <div v-if="store.error" class="text-sm text-red-600 mb-3">{{ store.error }}</div>

        <div v-if="store.latestPrediction" class="space-y-3">
          <div
            v-if="!store.latestPrediction.data_hash_match"
            class="text-xs px-3 py-2 bg-amber-50 border border-amber-200 text-amber-800 rounded"
          >
            ⚠ 학습 시점 데이터와 현재 <code>lotto.xlsx</code> 의 SHA-256 이 다릅니다.
            예측은 가능하지만 학습 분포와 어긋날 수 있습니다.
          </div>

          <div>
            <div class="text-xs text-slate-500 mb-1">LSTM 모델 예측</div>
            <NumberBalls :numbers="store.latestPrediction.lstm" />
          </div>

          <div>
            <div class="text-xs text-slate-500 mb-1">앙상블 예측</div>
            <NumberBalls :numbers="store.latestPrediction.ensemble" />
          </div>

          <div v-if="store.latestPrediction.additional_sets.length">
            <div class="text-xs text-slate-500 mb-1">추가 추천 세트</div>
            <div class="space-y-1.5">
              <NumberBalls
                v-for="(set, i) in store.latestPrediction.additional_sets"
                :key="i"
                :numbers="set"
              />
            </div>
          </div>

          <div class="text-xs text-slate-400 pt-1">
            sequence_length={{ store.latestPrediction.sequence_length }} · seed={{ store.latestPrediction.seed }}
          </div>
        </div>
      </template>
    </section>

    <!-- Recent draws -->
    <section class="bg-white border border-slate-200 rounded p-4">
      <h2 class="font-medium text-base mb-3">
        최근 회차
        <span class="text-xs text-slate-500">(총 {{ store.totalDraws }}회)</span>
      </h2>
      <ul class="divide-y divide-slate-100 text-sm">
        <li
          v-for="row in store.recentDraws"
          :key="row.draw_no"
          class="py-2 flex items-center gap-4"
        >
          <span class="text-xs text-slate-500 w-12">{{ row.draw_no }}회</span>
          <NumberBalls :numbers="row.numbers" small />
        </li>
      </ul>
    </section>
  </div>
</template>
