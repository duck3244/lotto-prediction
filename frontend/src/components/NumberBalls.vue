<script setup lang="ts">
defineProps<{
  numbers: number[]
  small?: boolean
}>()

function colorClass(n: number): string {
  if (n <= 10) return 'bg-amber-400'
  if (n <= 20) return 'bg-blue-400'
  if (n <= 30) return 'bg-rose-400'
  if (n <= 40) return 'bg-slate-500'
  return 'bg-emerald-500'
}
</script>

<template>
  <div class="flex gap-1.5">
    <span
      v-for="n in numbers"
      :key="n"
      :class="[
        'rounded-full text-white font-medium flex items-center justify-center select-none',
        colorClass(n),
        small ? 'w-7 h-7 text-xs' : 'w-9 h-9 text-sm',
      ]"
    >{{ n }}</span>
  </div>
</template>
