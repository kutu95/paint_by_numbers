import type { User } from '@supabase/supabase-js'
import { listMemberships } from '@/lib/auth/membership'
import { serviceSupabase } from '@/lib/auth/supabase-server'

export type ListedUser = {
  userId: string
  email: string
  role: 'admin' | 'user'
  displayName: string | null
  createdAt: string | null
  lastSignInAt: string | null
}

async function authUsersById(ids: string[]): Promise<Map<string, User>> {
  const wanted = new Set(ids)
  const found = new Map<string, User>()
  const supabase = serviceSupabase()
  for (let page = 1; page <= 20 && found.size < wanted.size; page += 1) {
    const { data, error } = await supabase.auth.admin.listUsers({ page, perPage: 200 })
    if (error) throw error
    for (const user of data.users) {
      if (wanted.has(user.id)) found.set(user.id, user)
    }
    if (data.users.length < 200) break
  }
  return found
}

export async function listLayerPainterUsers(): Promise<ListedUser[]> {
  const memberships = await listMemberships()
  const accounts = await authUsersById(memberships.map((membership) => membership.userId))
  return memberships
    .map((membership) => {
      const account = accounts.get(membership.userId)
      return {
        userId: membership.userId,
        email: account?.email || '',
        role: membership.role,
        displayName: membership.displayName,
        createdAt: account?.created_at || membership.createdAt,
        lastSignInAt: account?.last_sign_in_at || null,
      }
    })
    .sort((a, b) => a.email.localeCompare(b.email))
}
