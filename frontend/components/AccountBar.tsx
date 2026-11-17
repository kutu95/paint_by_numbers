'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { browserSupabase } from '@/lib/auth/supabase-browser'
import { clearProjectCache } from '@/lib/projects'

type Me = {
  email: string
  role: 'admin' | 'user'
}

export function AccountBar() {
  const authRequired = process.env.NEXT_PUBLIC_AUTH_REQUIRED === '1'
  const [me, setMe] = useState<Me | null>(null)

  useEffect(() => {
    if (!authRequired) return
    let active = true
    fetch('/api/auth/me', { credentials: 'include', cache: 'no-store' })
      .then(async (response) => (response.ok ? response.json() : null))
      .then((body) => {
        if (active && body?.email) setMe({ email: body.email, role: body.role === 'admin' ? 'admin' : 'user' })
      })
      .catch(() => {})
    return () => {
      active = false
    }
  }, [authRequired])

  if (!authRequired || !me) return null

  const logout = async () => {
    await fetch('/api/auth/logout', { method: 'POST', credentials: 'include' }).catch(() => {})
    try {
      await browserSupabase().auth.signOut()
    } catch {
      /* the shared login client may be unconfigured in local dev */
    }
    clearProjectCache()
    window.location.href = '/login'
  }

  return (
    <div className="flex items-center gap-3 text-sm text-gray-300">
      <span className="max-w-[16rem] truncate">{me.email}</span>
      {me.role === 'admin' ? <Link href="/manage-users" className="text-blue-300 hover:underline">Users</Link> : null}
      <button type="button" onClick={() => void logout()} className="rounded bg-gray-700 px-2 py-1 hover:bg-gray-600">Log out</button>
    </div>
  )
}
