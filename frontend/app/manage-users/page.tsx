'use client'

import { FormEvent, useCallback, useEffect, useState } from 'react'
import Link from 'next/link'

type ListedUser = {
  userId: string
  email: string
  role: 'admin' | 'user'
  displayName: string | null
  createdAt: string | null
  lastSignInAt: string | null
}

function formatWhen(value: string | null) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '—'
  return date.toLocaleString()
}

export default function ManageUsersPage() {
  const [users, setUsers] = useState<ListedUser[]>([])
  const [email, setEmail] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [role, setRole] = useState<'user' | 'admin'>('user')
  const [sendEmail, setSendEmail] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [embedded, setEmbedded] = useState(false)

  useEffect(() => {
    setEmbedded(new URLSearchParams(window.location.search).get('embed') === '1')
  }, [])

  const load = useCallback(async () => {
    const response = await fetch('/api/users', { cache: 'no-store', credentials: 'include' })
    const body = await response.json().catch(() => ({}))
    if (!response.ok) {
      throw new Error(body.error || 'Could not list users')
    }
    setUsers(Array.isArray(body.users) ? body.users : [])
  }, [])

  useEffect(() => {
    void load()
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setLoading(false))
  }, [load])

  const run = async (action: () => Promise<void>) => {
    setBusy(true)
    setError(null)
    setNotice(null)
    try {
      await action()
      await load()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Something went wrong')
    } finally {
      setBusy(false)
    }
  }

  const createUser = (event: FormEvent) => {
    event.preventDefault()
    void run(async () => {
      const response = await fetch('/api/users', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, displayName, role, sendEmail }),
      })
      const body = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(body.error || 'Could not create that account')
      setEmail('')
      setDisplayName('')
      setRole('user')
      setNotice(sendEmail ? 'Account created and login email sent.' : 'Account created.')
    })
  }

  const changeRole = (user: ListedUser, nextRole: 'admin' | 'user') => {
    void run(async () => {
      const response = await fetch('/api/users', {
        method: 'PATCH',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId: user.userId, role: nextRole }),
      })
      const body = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(body.error || 'Could not update the role')
    })
  }

  const sendLink = (user: ListedUser) => {
    void run(async () => {
      const response = await fetch('/api/users/login-link', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId: user.userId }),
      })
      const body = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(body.error || 'Could not send the login email')
      setNotice(`Login email sent to ${user.email}.`)
    })
  }

  const removeUser = (user: ListedUser) => {
    if (!window.confirm(`Remove LayerPainter access for ${user.email}? Their projects stay on disk.`)) return
    void run(async () => {
      const response = await fetch('/api/users', {
        method: 'DELETE',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId: user.userId }),
      })
      const body = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(body.error || 'Could not remove that account')
    })
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <header className="border-b border-gray-700 px-4 py-3 flex items-center justify-between">
        <h1 className="text-xl font-bold">Users</h1>
        {embedded ? null : <Link href="/" className="text-sm text-blue-300 hover:underline">Back to LayerPainter</Link>}
      </header>
      <main className="max-w-5xl mx-auto p-6 space-y-6">
        {error ? <p className="rounded bg-red-900/40 px-3 py-2 text-sm text-red-100">{error}</p> : null}
        {notice ? <p className="rounded bg-green-900/40 px-3 py-2 text-sm text-green-100">{notice}</p> : null}

        <form onSubmit={createUser} className="rounded-lg bg-gray-800 p-4 space-y-3">
          <h2 className="font-semibold">Add a user</h2>
          <div className="flex flex-wrap gap-3">
            <input required type="email" placeholder="Email" value={email} onChange={(event) => setEmail(event.target.value)} className="rounded bg-gray-700 px-3 py-2" />
            <input type="text" placeholder="Name" value={displayName} onChange={(event) => setDisplayName(event.target.value)} className="rounded bg-gray-700 px-3 py-2" />
            <select value={role} onChange={(event) => setRole(event.target.value === 'admin' ? 'admin' : 'user')} className="rounded bg-gray-700 px-3 py-2">
              <option value="user">Member</option>
              <option value="admin">Admin</option>
            </select>
            <label className="flex items-center gap-2 text-sm text-gray-300">
              <input type="checkbox" checked={sendEmail} onChange={(event) => setSendEmail(event.target.checked)} />
              Send login email
            </label>
            <button type="submit" disabled={busy} className="rounded bg-blue-600 px-4 py-2 font-medium hover:bg-blue-500 disabled:opacity-60">Add</button>
          </div>
          <p className="text-sm text-gray-400">A new person gets a login email and chooses their own password. An email that already signs in elsewhere keeps that password.</p>
        </form>

        {loading ? <p className="text-sm text-gray-400">Loading…</p> : (
          <div className="overflow-x-auto rounded-lg bg-gray-800">
            <table className="w-full text-sm">
              <thead className="text-left text-gray-400">
                <tr>
                  <th className="px-4 py-3">Email</th>
                  <th className="px-4 py-3">Name</th>
                  <th className="px-4 py-3">Role</th>
                  <th className="px-4 py-3">Last sign-in</th>
                  <th className="px-4 py-3">Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.map((user) => (
                  <tr key={user.userId} className="border-t border-gray-700">
                    <td className="px-4 py-3">{user.email || user.userId}</td>
                    <td className="px-4 py-3">{user.displayName || '—'}</td>
                    <td className="px-4 py-3 capitalize">{user.role}</td>
                    <td className="px-4 py-3">{formatWhen(user.lastSignInAt)}</td>
                    <td className="px-4 py-3">
                      <div className="flex flex-wrap gap-2">
                        {user.role === 'admin' ? (
                          <button type="button" disabled={busy} onClick={() => changeRole(user, 'user')} className="rounded bg-gray-700 px-2 py-1 hover:bg-gray-600 disabled:opacity-60">Make member</button>
                        ) : (
                          <button type="button" disabled={busy} onClick={() => changeRole(user, 'admin')} className="rounded bg-gray-700 px-2 py-1 hover:bg-gray-600 disabled:opacity-60">Make admin</button>
                        )}
                        <button type="button" disabled={busy} onClick={() => sendLink(user)} className="rounded bg-gray-700 px-2 py-1 hover:bg-gray-600 disabled:opacity-60">Send login link</button>
                        <button type="button" disabled={busy} onClick={() => removeUser(user)} className="rounded bg-red-900/80 px-2 py-1 text-red-100 hover:bg-red-800 disabled:opacity-60">Remove</button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </main>
    </div>
  )
}
