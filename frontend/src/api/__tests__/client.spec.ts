import axios, { AxiosError, AxiosHeaders } from 'axios'
import { describe, expect, it } from 'vitest'

import { asApiError } from '../client'

describe('asApiError', () => {
  it('extracts detail from FastAPI error body', () => {
    const headers = new AxiosHeaders()
    const err = new AxiosError(
      'Request failed with status code 400',
      'ERR_BAD_REQUEST',
      { headers } as never,
      undefined,
      {
        status: 400,
        statusText: 'Bad Request',
        headers: {},
        config: { headers } as never,
        data: { detail: '활성 모델이 없습니다.' },
      } as never,
    )
    expect(axios.isAxiosError(err)).toBe(true)
    const parsed = asApiError(err)
    expect(parsed.status).toBe(400)
    expect(parsed.detail).toBe('활성 모델이 없습니다.')
  })

  it('falls back to message when response is missing', () => {
    const err = new AxiosError('Network Error', 'ERR_NETWORK')
    const parsed = asApiError(err)
    expect(parsed.status).toBe(0)
    expect(parsed.detail).toBe('Network Error')
  })

  it('handles non-Axios errors', () => {
    const parsed = asApiError(new Error('boom'))
    expect(parsed.status).toBe(0)
    expect(parsed.detail).toBe('boom')
  })
})
