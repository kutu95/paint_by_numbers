'use client'

import { FormEvent, Suspense, useState } from 'react'
import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { AuthCard } from '@/components/AuthCard'
import { browserSupabase } from '@/lib/auth/supabase-browser'
import { clearProjectCache } from '@/lib/projects'

function safeNext(value: string | null): string {
  if (!value || !value.startsWith('/') || value.startsWith('//')) return '/'
  return value
}

function LoginForm() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const supabase = browserSupabase()
      const { error: signInError } = await supabase.auth.signInWithPassword({
        email: email.trim().toLowerCase(),
        password,
      })
      if (signInError) {
        setError('The email or password is incorrect.')
        return
      }
      const response = await fetch('/api/auth/session', { method: 'POST', credentials: 'include' })
      const body = await response.json().catch(() => ({}))
      if (!response.ok) {
        await supabase.auth.signOut()
        setError(body.error || 'This email does not have a LayerPainter account.')
        return
      }
      clearProjectCache()
      router.push(safeNext(searchParams.get('redirectTo')))
      router.refresh()
    } catch {
      setError('Sign in is not available. Check that the login service is configured.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuthCard title="Sign in" subtitle="Use the password for this email. It is the same login as your other apps.">
      <form className="space-y-4" onSubmit={handleSubmit}>
        {error ? <p className="rounded bg-red-900/40 px-3 py-2 text-sm text-red-100">{error}</p> : null}
        <label className="block text-sm text-gray-300">
          Email
          <input
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            className="mt-1 w-full rounded bg-gray-700 px-3 py-2 text-white"
          />
        </label>
        <label className="block text-sm text-gray-300">
          Password
          <input
            type="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            className="mt-1 w-full rounded bg-gray-700 px-3 py-2 text-white"
          />
        </label>
        <button type="submit" disabled={loading} className="w-full rounded bg-blue-600 px-4 py-2 font-medium hover:bg-blue-500 disabled:opacity-60">
          {loading ? 'Signing in…' : 'Sign in'}
        </button>
      </form>
      <p className="mt-4 text-center text-sm">
        <Link href="/forgot-password" className="text-blue-300 hover:underline">Forgot your password?</Link>
      </p>
    </AuthCard>
  )
}

export default function LoginPage() {
  return (
    <Suspense fallback={<AuthCard title="Sign in"><p className="text-sm text-gray-400">Loading…</p></AuthCard>}>
      <LoginForm />
    </Suspense>
  )
}
