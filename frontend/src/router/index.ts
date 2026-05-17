import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'dashboard',
      component: () => import('../views/DashboardView.vue'),
    },
    {
      path: '/models',
      name: 'models',
      component: () => import('../views/ModelsView.vue'),
    },
    {
      path: '/train',
      name: 'train',
      component: () => import('../views/TrainView.vue'),
    },
    {
      path: '/stats',
      name: 'stats',
      component: () => import('../views/StatsView.vue'),
    },
  ],
})

export default router
