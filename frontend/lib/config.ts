// API configuration
// In Next.js, environment variables prefixed with NEXT_PUBLIC_ are exposed to the browser
const envBase = process.env.NEXT_PUBLIC_API_BASE_URL || ''

function getDefaultApiBase(): string {
  // In local dev, bypass Next.js proxy to avoid intermittent proxy socket resets.
  if (typeof window !== 'undefined') {
    const host = window.location.hostname
    if (host === 'localhost' || host === '127.0.0.1') {
      return 'http://127.0.0.1:8000'
    }
  }
  return ''
}

const rawBase = envBase || getDefaultApiBase()
export const API_BASE_URL = rawBase.endsWith('/') ? rawBase.slice(0, -1) : rawBase
