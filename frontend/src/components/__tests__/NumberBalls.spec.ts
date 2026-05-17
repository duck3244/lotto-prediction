import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'

import NumberBalls from '../NumberBalls.vue'

describe('NumberBalls', () => {
  it('renders one span per number', () => {
    const wrapper = mount(NumberBalls, { props: { numbers: [3, 14, 25, 36, 41] } })
    const spans = wrapper.findAll('span')
    expect(spans).toHaveLength(5)
    expect(spans.map((s) => s.text())).toEqual(['3', '14', '25', '36', '41'])
  })

  it('applies color class by number range', () => {
    const wrapper = mount(NumberBalls, { props: { numbers: [5, 15, 25, 35, 45] } })
    const classes = wrapper.findAll('span').map((s) => s.classes())
    expect(classes[0]).toContain('bg-amber-400')   // 1-10
    expect(classes[1]).toContain('bg-blue-400')    // 11-20
    expect(classes[2]).toContain('bg-rose-400')    // 21-30
    expect(classes[3]).toContain('bg-slate-500')   // 31-40
    expect(classes[4]).toContain('bg-emerald-500') // 41-45
  })

  it('switches to small size when small prop is set', () => {
    const wrapper = mount(NumberBalls, { props: { numbers: [1], small: true } })
    expect(wrapper.find('span').classes()).toContain('w-7')
  })
})
