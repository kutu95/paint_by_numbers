import { createServerClient, type CookieOptions } from '@supabase/ssr'
import { createClient, type SupabaseClient } from '@supabase/supabase-js'
import { cookies } from 'next/headers'

const publicAuthUrl = () => (process.env.NEXT_PUBLIC_SUPABASE_URL || '').replace(/\/$/, '')
const internalAuthUrl = () => (process.env.SUPABASE_INTERNAL_URL || '').replace(/\/$/, '')

/** Keep the public project URL for cookie names, but call GoTrue on this machine. */
export function serverAuthFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const pub = publicAuthUrl()
  const internal = internalAuthUrl()
  const rewrite = (href: string) =>
    pub && internal && href.startsWith(pub) ? internal + href.slice(pub.length) : href
  if (typeof input === 'string') return fetch(rewrite(input), init)
  if (input instanceof URL) return fetch(rewrite(input.toString()), init)
  const next = rewrite(input.url)
  if (next === input.url) return fetch(input, init)
  return fetch(new Request(next, input), init)
}

export function routeSupabase() {
  const cookieStore = cookies()
  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL || '',
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '',
    {
      global: { fetch: serverAuthFetch },
      cookies: {
        get(name: string) {
          return cookieStore.get(name)?.value
        },
        set(name: string, value: string, options: CookieOptions) {
          cookieStore.set({ name, value, ...options })
        },
        remove(name: string, options: CookieOptions) {
          cookieStore.set({ name, value: '', ...options, maxAge: 0 })
        },
      },
    },
  )
}

export function serviceSupabase(): SupabaseClient {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY
  if (!url || !key) {
    throw new Error('Supabase admin is not configured')
  }
  return createClient(url, key, {
    global: { fetch: serverAuthFetch },
    auth: { autoRefreshToken: false, persistSession: false },
  })
}
