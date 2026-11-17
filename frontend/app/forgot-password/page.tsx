'use client'

import { FormEvent, useState } from 'react'
import Link from 'next/link'
import { AuthCard } from '@/components/AuthCard'

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/auth/forgot-password', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email }),
      })
      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        setError(body.error || 'Could not send the reset email. Try again later.')
        return
      }
      setSuccess(true)
    } catch {
      setError('Could not send the reset email. Try again later.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuthCard title="Reset your password" subtitle="We'll email you a link to choose a new password.">
      {success ? (
        <p className="rounded bg-green-900/40 px-3 py-2 text-sm text-green-100">Check your email for a link to reset your password.</p>
      ) : (
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
          <button type="submit" disabled={loading} className="w-full rounded bg-blue-600 px-4 py-2 font-medium hover:bg-blue-500 disabled:opacity-60">
            {loading ? 'Sending reset link…' : 'Send reset link'}
          </button>
        </form>
      )}
      <p className="mt-4 text-center text-sm">
        <Link href="/login" className="text-blue-300 hover:underline">Back to sign in</Link>
      </p>
    </AuthCard>
  )
}
