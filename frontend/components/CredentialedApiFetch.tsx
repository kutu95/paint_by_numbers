'use client'

import { useEffect } from 'react'

export function CredentialedApiFetch() {
  useEffect(() => {
    const original = window.fetch.bind(window)
    window.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
      if (url.includes('/api/')) {
        return original(input, { ...init, credentials: init?.credentials ?? 'include' })
      }
      return original(input, init)
    }) as typeof window.fetch
    return () => {
      window.fetch = original
    }
  }, [])
  return null
}
