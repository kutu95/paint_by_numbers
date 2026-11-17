'use client'

import { FormEvent, useCallback, useEffect, useState } from 'react'

type Collaborator = {
  userId: string
  email: string
  displayName: string | null
}

export function ProjectCollaborators({ projectId, isOwner }: { projectId: string; isOwner: boolean }) {
  const [collaborators, setCollaborators] = useState<Collaborator[]>([])
  const [email, setEmail] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const load = useCallback(async () => {
    const response = await fetch(`/api/projects/${encodeURIComponent(projectId)}/collaborators`, {
      credentials: 'include',
      cache: 'no-store',
    })
    const body = await response.json().catch(() => ({}))
    if (!response.ok) throw new Error(body.error || 'Could not load collaborators')
    setCollaborators(Array.isArray(body.collaborators) ? body.collaborators : [])
  }, [projectId])

  useEffect(() => {
    if (process.env.NEXT_PUBLIC_AUTH_REQUIRED !== '1') return
    void load().catch((reason: Error) => setError(reason.message))
  }, [load])

  if (process.env.NEXT_PUBLIC_AUTH_REQUIRED !== '1') return null

  const invite = (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError(null)
    setNotice(null)
    void (async () => {
      try {
        const response = await fetch(`/api/projects/${encodeURIComponent(projectId)}/collaborators`, {
          method: 'POST',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email }),
        })
        const body = await response.json().catch(() => ({}))
        if (!response.ok) throw new Error(body.error || 'Could not invite that person')
        setCollaborators(Array.isArray(body.collaborators) ? body.collaborators : [])
        setEmail('')
        setNotice('They can now open this project from their own list.')
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : 'Could not invite that person')
      } finally {
        setBusy(false)
      }
    })()
  }

  const remove = (collaborator: Collaborator) => {
    setBusy(true)
    setError(null)
    setNotice(null)
    void (async () => {
      try {
        const response = await fetch(`/api/projects/${encodeURIComponent(projectId)}/collaborators`, {
          method: 'DELETE',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ userId: collaborator.userId }),
        })
        const body = await response.json().catch(() => ({}))
        if (!response.ok) throw new Error(body.error || 'Could not remove that person')
        setCollaborators(Array.isArray(body.collaborators) ? body.collaborators : [])
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : 'Could not remove that person')
      } finally {
        setBusy(false)
      }
    })()
  }

  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800/60 p-4 space-y-3">
      <h2 className="text-sm font-semibold text-gray-300">Collaborators</h2>
      {error ? <p className="text-sm text-red-200">{error}</p> : null}
      {notice ? <p className="text-sm text-green-200">{notice}</p> : null}
      {collaborators.length === 0 ? (
        <p className="text-sm text-gray-500">{isOwner ? 'Only you can see this project.' : 'You are collaborating on this project.'}</p>
      ) : (
        <ul className="space-y-2">
          {collaborators.map((collaborator) => (
            <li key={collaborator.userId} className="flex items-center justify-between gap-3 text-sm">
              <span className="truncate">{collaborator.displayName || collaborator.email || collaborator.userId}</span>
              {isOwner ? (
                <button type="button" disabled={busy} onClick={() => remove(collaborator)} className="shrink-0 text-gray-400 hover:text-white disabled:opacity-60">
                  Remove
                </button>
              ) : null}
            </li>
          ))}
        </ul>
      )}
      {isOwner ? (
        <form onSubmit={invite} className="flex flex-wrap gap-2">
          <input
            type="email"
            required
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            placeholder="Email of a LayerPainter user"
            className="min-w-[16rem] flex-1 rounded bg-gray-700 px-3 py-2 text-sm"
          />
          <button type="submit" disabled={busy} className="rounded bg-blue-600 px-3 py-2 text-sm font-medium hover:bg-blue-500 disabled:opacity-60">
            Invite
          </button>
        </form>
      ) : null}
    </div>
  )
}
