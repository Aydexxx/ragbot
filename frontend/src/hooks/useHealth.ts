import { useCallback, useEffect, useState } from 'react'
import { ApiError, getHealth } from '../api/client'
import type { HealthResponse } from '../api/types'

interface HealthState {
  data: HealthResponse | null
  loading: boolean
  error: string | null
}

export function useHealth() {
  const [state, setState] = useState<HealthState>({
    data: null,
    loading: true,
    error: null,
  })

  // No synchronous setState before this — only inside the .then()/.catch()
  // callbacks — so this is safe to call directly from the mount effect below.
  const load = useCallback(() => {
    return getHealth()
      .then((data) => setState({ data, loading: false, error: null }))
      .catch((err: unknown) => {
        const message =
          err instanceof ApiError ? err.message : 'Could not load provider status.'
        setState({ data: null, loading: false, error: message })
      })
  }, [])

  const refresh = useCallback(() => {
    setState((prev) => ({ ...prev, loading: true, error: null }))
    void load()
  }, [load])

  useEffect(() => {
    void load()
  }, [load])

  return { ...state, refresh }
}
