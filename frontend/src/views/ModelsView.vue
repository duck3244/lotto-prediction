<script setup lang="ts">
import { onMounted, ref } from 'vue'

import { asApiError } from '../api/client'
import { useAppStore } from '../stores/app'

const store = useAppStore()
const busy = ref<string | null>(null)
const message = ref<string | null>(null)

onMounted(() => {
  store.fetchModels()
})

async function activate(name: string) {
  busy.value = name
  message.value = null
  try {
    const r = await store.activateModel(name)
    message.value = r.message
  } catch (e) {
    message.value = asApiError(e).detail
  } finally {
    busy.value = null
  }
}
</script>

<template>
  <div class="space-y-4">
    <section class="bg-white border border-slate-200 rounded p-4">
      <h2 class="font-medium text-base mb-3">학습 번들</h2>

      <div v-if="!store.models.length" class="text-sm text-slate-600">
        등록된 번들이 없습니다.
        <code class="text-xs bg-slate-100 px-1.5 py-0.5 rounded">cd backend && python main.py</code>
        로 한 번 학습하면 <code class="text-xs bg-slate-100 px-1.5 py-0.5 rounded">models/bundle_&lt;ts&gt;/</code>
        가 자동 생성됩니다.
      </div>

      <table v-else class="w-full text-sm">
        <thead class="text-left text-xs text-slate-500 border-b border-slate-200">
          <tr>
            <th class="py-2 pr-3">이름</th>
            <th class="py-2 pr-3">학습 시점</th>
            <th class="py-2 pr-3 text-right">seq</th>
            <th class="py-2 pr-3">데이터 해시</th>
            <th class="py-2 pr-3">TF</th>
            <th class="py-2"></th>
          </tr>
        </thead>
        <tbody class="divide-y divide-slate-100">
          <tr v-for="b in store.models" :key="b.name">
            <td class="py-2 pr-3 font-mono text-xs">{{ b.name }}</td>
            <td class="py-2 pr-3 text-slate-500">{{ b.timestamp ?? '—' }}</td>
            <td class="py-2 pr-3 text-right">{{ b.sequence_length ?? '—' }}</td>
            <td class="py-2 pr-3">
              <span
                v-if="b.data_hash_match"
                class="inline-flex items-center gap-1 text-xs text-emerald-700 bg-emerald-50 border border-emerald-200 rounded px-1.5 py-0.5"
              >match</span>
              <span
                v-else
                class="inline-flex items-center gap-1 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-1.5 py-0.5"
              >mismatch</span>
            </td>
            <td class="py-2 pr-3 text-slate-500">{{ b.tensorflow_version ?? '—' }}</td>
            <td class="py-2 text-right">
              <span v-if="b.is_active" class="text-xs text-blue-600 font-medium">활성</span>
              <button
                v-else
                @click="activate(b.name)"
                :disabled="busy === b.name"
                class="px-2.5 py-1 text-xs bg-slate-100 border border-slate-300 rounded hover:bg-slate-200 disabled:opacity-50"
              >{{ busy === b.name ? '활성화 중…' : '활성화' }}</button>
            </td>
          </tr>
        </tbody>
      </table>

      <div
        v-if="message"
        class="mt-3 text-xs text-slate-700 bg-slate-50 border border-slate-200 rounded p-2"
      >{{ message }}</div>
    </section>
  </div>
</template>
