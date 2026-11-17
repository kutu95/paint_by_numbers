'use client'

import { FormEvent, Suspense, useEffect, useState } from 'react'
import Link from 'next/link'
import { useRouter, useSearchParams } from 'next/navigation'
import { AuthCard } from '@/components/AuthCard'
import { clearProjectCache } from '@/lib/projects'

const RECOVERY_FLAG = 'layerpainter-recovery'

function ResetPasswordForm() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [checkingSession, setCheckingSession] = useState(true)
  const [confirming, setConfirming] = useState(false)
  const [hasSession, setHasSession] = useState(false)
  const tokenHash = searchParams.get('token_hash')
  const hasRecoveryToken = Boolean(tokenHash && searchParams.get('type') === 'recovery' && tokenHash.length <= 2048)

  useEffect(() => {
    const confirmed = window.sessionStorage.getItem(RECOVERY_FLAG) === '1'
    if (hasRecoveryToken && !confirmed) {
      setHasSession(false)
      setCheckingSession(false)
      return
    }
    setHasSession(confirmed)
    setCheckingSession(false)
  }, [hasRecoveryToken])

  const confirmReset = async () => {
    if (!tokenHash) return
    setConfirming(true)
    setError(null)
    try {
      const response = await fetch('/api/auth/confirm-recovery', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token_hash: tokenHash }),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        setError(body.error || 'This reset link is invalid or has expired.')
        return
      }
      window.sessionStorage.setItem(RECOVERY_FLAG, '1')
      const url = new URL(window.location.href)
      url.searchParams.delete('token_hash')
      url.searchParams.delete('type')
      window.history.replaceState(null, '', `${url.pathname}${url.search}`)
      setHasSession(true)
    } catch {
      setError('This reset link is invalid or has expired.')
    } finally {
      setConfirming(false)
    }
  }

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    setError(null)
    if (password.length < 6) {
      setError('Password must be at least 6 characters.')
      return
    }
    if (password !== confirmPassword) {
      setError('Passwords do not match')
      return
    }
    setLoading(true)
    try {
      const response = await fetch('/api/auth/update-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password }),
      })
      const result = await response.json().catch(() => ({}))
      if (!response.ok) {
        setError(result.error || 'Could not update the password. Request a new reset link and try again.')
        return
      }
      window.sessionStorage.removeItem(RECOVERY_FLAG)
      clearProjectCache()
      router.push('/')
      router.refresh()
    } catch {
      setError('Could not update the password. Request a new reset link and try again.')
    } finally {
      setLoading(false)
    }
  }

  const subtitle = !checkingSession && !hasSession && hasRecoveryToken
    ? 'Opening the email does not use the link until you continue.'
    : 'Choose a new password. This replaces the password for this email on every app that shares the login.'

  return (
    <AuthCard title="Choose a password" subtitle={subtitle}>
      {checkingSession ? <p className="text-sm text-gray-400">Loading…</p> : null}
      {!checkingSession && !hasSession && hasRecoveryToken ? (
        <div className="space-y-4">
          {error ? <p className="rounded bg-red-900/40 px-3 py-2 text-sm text-red-100">{error}</p> : null}
          <button type="button" disabled={confirming} onClick={confirmReset} className="w-full rounded bg-blue-600 px-4 py-2 font-medium hover:bg-blue-500 disabled:opacity-60">
            {confirming ? 'Continuing…' : 'Continue'}
          </button>
          {error ? <Link href="/forgot-password" className="block text-center text-sm text-blue-300 hover:underline">Send a new reset link</Link> : null}
        </div>
      ) : null}
      {!checkingSession && !hasSession && !hasRecoveryToken ? (
        <div className="space-y-4">
          <p className="rounded bg-red-900/40 px-3 py-2 text-sm text-red-100">This reset link is missing or has expired. Request a new one.</p>
          <Link href="/forgot-password" className="block text-center rounded bg-blue-600 px-4 py-2 font-medium hover:bg-blue-500">Send a new reset link</Link>
        </div>
      ) : null}
      {!checkingSession && hasSession ? (
        <form className="space-y-4" onSubmit={handleSubmit}>
          {error ? <p className="rounded bg-red-900/40 px-3 py-2 text-sm text-red-100">{error}</p> : null}
          <label className="block text-sm text-gray-300">
            New password
            <input type="password" autoComplete="new-password" required value={password} onChange={(event) => setPassword(event.target.value)} className="mt-1 w-full rounded bg-gray-700 px-3 py-2 text-white" />
          </label>
          <label className="block text-sm text-gray-300">
            Confirm password
            <input type="password" autoComplete="new-password" required value={confirmPassword} onChange={(event) => setConfirmPassword(event.target.value)} className="mt-1 w-full rounded bg-gray-700 px-3 py-2 text-white" />
          </label>
          <button type="submit" disabled={loading} className="w-full rounded bg-blue-600 px-4 py-2 font-medium hover:bg-blue-500 disabled:opacity-60">
            {loading ? 'Saving…' : 'Save password'}
          </button>
        </form>
      ) : null}
    </AuthCard>
  )
}

export default function ResetPasswordPage() {
  return (
    <Suspense fallback={<AuthCard title="Choose a password"><p className="text-sm text-gray-400">Loading…</p></AuthCard>}>
      <ResetPasswordForm />
    </Suspense>
  )
}
