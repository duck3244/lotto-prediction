<script setup lang="ts">
import { onMounted } from 'vue'
import { RouterLink, RouterView } from 'vue-router'

import { useAppStore } from './stores/app'

const store = useAppStore()

onMounted(() => {
  store.fetchHealth()
})
</script>

<template>
  <div class="min-h-screen flex flex-col">
    <header class="bg-white border-b border-slate-200">
      <div class="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
        <div class="flex items-center gap-6">
          <h1 class="text-lg font-semibold">로또 예측</h1>
          <nav class="flex gap-3 text-sm">
            <RouterLink to="/" exact-active-class="text-blue-600 font-medium"
                        class="text-slate-600 hover:text-slate-900">대시보드</RouterLink>
            <RouterLink to="/models" exact-active-class="text-blue-600 font-medium"
                        class="text-slate-600 hover:text-slate-900">모델</RouterLink>
            <RouterLink to="/train" exact-active-class="text-blue-600 font-medium"
                        class="text-slate-600 hover:text-slate-900">학습</RouterLink>
            <RouterLink to="/stats" exact-active-class="text-blue-600 font-medium"
                        class="text-slate-600 hover:text-slate-900">통계</RouterLink>
          </nav>
        </div>
        <div v-if="store.health" class="text-xs text-slate-500">
          TF {{ store.health.tensorflow_version }} · GPU
          <span :class="store.health.gpu_available ? 'text-emerald-600' : 'text-slate-400'">
            {{ store.health.gpu_available ? '✓' : '×' }}
          </span>
        </div>
      </div>
    </header>

    <main class="flex-1 max-w-6xl mx-auto w-full px-4 py-6">
      <RouterView />
    </main>

    <footer class="border-t border-slate-200 bg-white">
      <div class="max-w-6xl mx-auto px-4 py-3 text-xs text-slate-500">
        ⚠ 본 도구는 학습/오락 목적입니다. 실제 당첨과 무관하며, 책임 있는 복권 참여를 권장합니다.
      </div>
    </footer>
  </div>
</template>
