import { randomBytes } from 'crypto'
import { insertMembership, membershipFor } from '@/lib/auth/membership'
import { findAuthUserByEmail } from '@/lib/auth/password-reset'
import { serviceSupabase } from '@/lib/auth/supabase-server'

export async function provisionMember(email: string): Promise<{ userId: string; created: boolean }> {
  let account = await findAuthUserByEmail(email)
  if (!account) {
    const { data, error } = await serviceSupabase().auth.admin.createUser({
      email,
      password: randomBytes(32).toString('base64url'),
      email_confirm: true,
      user_metadata: { app: 'layerpainter' },
    })
    if (error || !data.user) {
      console.error('Could not create the login:', error?.message || 'missing user')
      throw new Error('Could not create that account')
    }
    account = data.user
  }

  const existing = await membershipFor(account.id)
  if (existing) return { userId: account.id, created: false }
  const inserted = await insertMembership(account.id, 'user', null)
  if (!inserted && !(await membershipFor(account.id))) {
    throw new Error('Could not create that account')
  }
  return { userId: account.id, created: true }
}
