'use client'

import { FormEvent, useCallback, useEffect, useId, useMemo, useState } from 'react'

type Collaborator = {
  userId: string
  email: string
  displayName: string | null
}

type DirectoryUser = {
  userId: string
  email: string
  displayName: string | null
}

function looksLikeEmail(value: string) {
  return value.length <= 320 && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)
}

function labelFor(user: { email: string; displayName: string | null }) {
  if (user.displayName && user.email) return `${user.displayName} · ${user.email}`
  return user.displayName || user.email
}

export function ProjectCollaborators({ projectId, isOwner, isAdmin }: { projectId: string; isOwner: boolean; isAdmin: boolean }) {
  const listId = useId()
  const [collaborators, setCollaborators] = useState<Collaborator[]>([])
  const [directory, setDirectory] = useState<DirectoryUser[]>([])
  const [query, setQuery] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [open, setOpen] = useState(false)
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
    setDirectory(Array.isArray(body.directory) ? body.directory : [])
  }, [projectId])

  useEffect(() => {
    if (process.env.NEXT_PUBLIC_AUTH_REQUIRED !== '1') return
    void load().catch((reason: Error) => setError(reason.message))
  }, [load])

  const collaboratorIds = useMemo(() => new Set(collaborators.map((person) => person.userId)), [collaborators])
  const typed = query.trim().toLowerCase()
  const available = directory.filter((person) => !collaboratorIds.has(person.userId))
  const matches = available.filter((person) => {
    if (!typed) return true
    return person.email.toLowerCase().includes(typed) || (person.displayName || '').toLowerCase().includes(typed)
  })
  const selected = available.find((person) => person.userId === selectedId) || null
  const exact = available.find((person) => person.email.toLowerCase() === typed) || null
  const knownEmail = directory.some((person) => person.email.toLowerCase() === typed)
    || collaborators.some((person) => person.email.toLowerCase() === typed)
  const alreadyAdded = collaborators.find((person) => person.email.toLowerCase() === typed) || null
  const canCreate = looksLikeEmail(typed) && !knownEmail && !alreadyAdded
  const inviteTarget = selected && selected.email.toLowerCase() === typed ? selected : exact

  if (process.env.NEXT_PUBLIC_AUTH_REQUIRED !== '1') return null

  const invite = (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError(null)
    setNotice(null)
    const body = inviteTarget
      ? { userId: inviteTarget.userId }
      : { email: typed }
    void (async () => {
      try {
        const response = await fetch(`/api/projects/${encodeURIComponent(projectId)}/collaborators`, {
          method: 'POST',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })
        const payload = await response.json().catch(() => ({}))
        if (Array.isArray(payload.collaborators)) setCollaborators(payload.collaborators)
        if (!response.ok) throw new Error(payload.error || 'Could not invite that person')
        setQuery('')
        setSelectedId(null)
        setOpen(false)
        setNotice(payload.created
          ? 'Account created. They were emailed a link to choose a password and can open this project.'
          : 'They can now open this project from their own list.')
        await load()
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

  const choose = (person: DirectoryUser) => {
    setQuery(person.email)
    setSelectedId(person.userId)
    setOpen(false)
  }

  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800/60 p-4 space-y-3">
      <h2 className="text-sm font-semibold text-gray-300">Collaborators</h2>
      {error ? <p className="text-sm text-red-200">{error}</p> : null}
      {notice ? <p className="text-sm text-green-200">{notice}</p> : null}
      {collaborators.length === 0 ? (
        <p className="text-sm text-gray-500">
          {isOwner
            ? 'Only you can see this project.'
            : isAdmin
              ? 'No collaborators on this project.'
              : 'You are collaborating on this project.'}
        </p>
      ) : (
        <ul className="space-y-2">
          {collaborators.map((collaborator) => (
            <li key={collaborator.userId} className="flex items-center justify-between gap-3 text-sm">
              <span className="truncate">{labelFor(collaborator)}</span>
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
        <form onSubmit={invite} className="space-y-2">
          <div className="flex flex-wrap gap-2">
            <div className="relative min-w-[16rem] flex-1">
              <input
                type="text"
                role="combobox"
                aria-expanded={open}
                aria-controls={listId}
                aria-autocomplete="list"
                required
                value={query}
                onChange={(event) => {
                  setQuery(event.target.value)
                  setSelectedId(null)
                  setOpen(true)
                }}
                onFocus={() => setOpen(true)}
                onBlur={() => {
                  window.setTimeout(() => setOpen(false), 150)
                }}
                placeholder="Choose a user or type a new email"
                className="w-full rounded bg-gray-700 px-3 py-2 text-sm"
              />
              {open && (matches.length > 0 || canCreate) ? (
                <ul id={listId} role="listbox" className="absolute z-20 mt-1 max-h-60 w-full overflow-auto rounded-lg border border-gray-600 bg-gray-800 py-1 shadow-lg">
                  {matches.map((person) => (
                    <li key={person.userId}>
                      <button
                        type="button"
                        role="option"
                        aria-selected={person.userId === selectedId}
                        onMouseDown={(event) => event.preventDefault()}
                        onClick={() => choose(person)}
                        className="block w-full px-3 py-2 text-left text-sm hover:bg-gray-700"
                      >
                        {labelFor(person)}
                      </button>
                    </li>
                  ))}
                  {canCreate ? (
                    <li>
                      <button
                        type="button"
                        onMouseDown={(event) => event.preventDefault()}
                        onClick={() => setOpen(false)}
                        className="block w-full px-3 py-2 text-left text-sm text-blue-200 hover:bg-gray-700"
                      >
                        Create {typed} and email them a login link
                      </button>
                    </li>
                  ) : null}
                </ul>
              ) : null}
            </div>
            <button
              type="submit"
              disabled={busy || (!inviteTarget && !canCreate)}
              className="rounded bg-blue-600 px-3 py-2 text-sm font-medium hover:bg-blue-500 disabled:opacity-60"
            >
              {canCreate && !inviteTarget ? 'Create and invite' : 'Invite'}
            </button>
          </div>
          {alreadyAdded ? (
            <p className="text-sm text-gray-400">{labelFor(alreadyAdded)} is already on this project.</p>
          ) : (
            <p className="text-sm text-gray-500">
              Pick someone who already has an account, or type a new email. A new person is added as a user, emailed a link to choose a password, and added to this project.
            </p>
          )}
        </form>
      ) : null}
    </div>
  )
}
