<script setup lang="ts">
import { computed, onMounted } from 'vue'
import {
  ArcElement,
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  Title,
  Tooltip,
} from 'chart.js'
import { Bar, Doughnut } from 'vue-chartjs'

import { useAppStore } from '../stores/app'

ChartJS.register(CategoryScale, LinearScale, BarElement, ArcElement, Title, Tooltip, Legend)

const store = useAppStore()

onMounted(() => {
  if (!store.stats) store.fetchStats()
})

function colorFor(n: number): string {
  if (n <= 10) return '#fbbf24'
  if (n <= 20) return '#60a5fa'
  if (n <= 30) return '#fb7185'
  if (n <= 40) return '#64748b'
  return '#10b981'
}

const frequencyData = computed(() => {
  const stats = store.stats
  if (!stats) return null
  return {
    labels: stats.frequencies.map((f) => f.number.toString()),
    datasets: [
      {
        label: '출현 횟수',
        data: stats.frequencies.map((f) => f.count),
        backgroundColor: stats.frequencies.map((f) => colorFor(f.number)),
      },
    ],
  }
})

const frequencyOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { display: false } },
  scales: {
    x: { ticks: { autoSkip: false, maxRotation: 0 } },
  },
}

const oddEvenData = computed(() => {
  const stats = store.stats
  if (!stats) return null
  const keys = Object.keys(stats.odd_even).sort()
  return {
    labels: keys,
    datasets: [
      {
        data: keys.map((k) => stats.odd_even[k]),
        backgroundColor: ['#60a5fa', '#fb7185', '#fbbf24', '#10b981', '#64748b', '#a78bfa', '#34d399'],
      },
    ],
  }
})

const rangeBuckets = ['1-10', '11-20', '21-30', '31-40', '41-45']

const rangeData = computed(() => {
  const stats = store.stats
  if (!stats) return null
  // backend 의 range_distribution 키는 "n:n:n:n:n" 패턴별 빈도. 구간별 총합으로 환산.
  const totals = [0, 0, 0, 0, 0]
  for (const [pattern, freq] of Object.entries(stats.range_distribution)) {
    const parts = pattern.split(':').map((x) => parseInt(x, 10))
    for (let i = 0; i < Math.min(parts.length, totals.length); i++) {
      totals[i] += parts[i] * freq
    }
  }
  return {
    labels: rangeBuckets,
    datasets: [
      {
        label: '구간 내 번호 출현 횟수',
        data: totals,
        backgroundColor: ['#fbbf24', '#60a5fa', '#fb7185', '#64748b', '#10b981'],
      },
    ],
  }
})
</script>

<template>
  <div class="space-y-4">
    <section class="bg-white border border-slate-200 rounded p-4">
      <div class="flex items-center justify-between mb-3">
        <h2 class="font-medium text-base">번호 출현 빈도</h2>
        <span v-if="store.stats" class="text-xs text-slate-500">
          전체 {{ store.stats.total_draws }}회차 기준
        </span>
      </div>
      <div v-if="frequencyData" class="h-72">
        <Bar :data="frequencyData" :options="frequencyOptions" />
      </div>
      <div v-else class="text-sm text-slate-500">통계 로드 중…</div>
    </section>

    <div class="grid md:grid-cols-2 gap-4">
      <section class="bg-white border border-slate-200 rounded p-4">
        <h2 class="font-medium text-base mb-3">홀짝 조합 분포</h2>
        <div v-if="oddEvenData" class="h-72">
          <Doughnut
            :data="oddEvenData"
            :options="{ responsive: true, maintainAspectRatio: false }"
          />
        </div>
        <div v-else class="text-sm text-slate-500">—</div>
      </section>

      <section class="bg-white border border-slate-200 rounded p-4">
        <h2 class="font-medium text-base mb-3">번호 구간별 출현</h2>
        <div v-if="rangeData" class="h-72">
          <Bar
            :data="rangeData"
            :options="{ responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }"
          />
        </div>
        <div v-else class="text-sm text-slate-500">—</div>
      </section>
    </div>
  </div>
</template>
