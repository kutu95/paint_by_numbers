import { NextResponse } from 'next/server'
import { cookies } from 'next/headers'
import { membershipFor } from '@/lib/auth/membership'
import { COOKIE_NAME, readSession, sessionCookieOptions, signSession, SESSION_TTL_SECONDS } from '@/lib/auth/session-cookie'

export type SignedInMember = {
  id: string
  email: string
  role: 'admin' | 'user'
  displayName: string | null
}

export async function readMember(): Promise<SignedInMember | null> {
  const secret = process.env.AUTH_SECRET || ''
  const token = cookies().get(COOKIE_NAME)?.value || ''
  const payload = await readSession(token, secret)
  if (!payload) return null
  const membership = await membershipFor(payload.uid)
  if (!membership) return null
  return {
    id: payload.uid,
    email: payload.email,
    role: membership.role,
    displayName: membership.displayName,
  }
}

export function memberError(status: 401 | 403) {
  const message = status === 401 ? 'Sign in required' : 'Admin access required'
  return NextResponse.json({ error: message }, { status })
}

export async function requireMember() {
  try {
    const member = await readMember()
    if (!member) return { error: memberError(401) }
    return { member }
  } catch (error) {
    console.error('Could not read the signed-in member:', error instanceof Error ? error.message : 'unknown')
    return { error: NextResponse.json({ error: 'Account directory is unavailable' }, { status: 503 }) }
  }
}

export async function requireAdmin() {
  const result = await requireMember()
  if (result.error || !result.member) return result
  if (result.member.role !== 'admin') return { error: memberError(403) }
  return result
}

export async function withSessionCookie(response: NextResponse, user: { id: string; email: string }) {
  const secret = process.env.AUTH_SECRET || ''
  if (secret.length < 16) {
    throw new Error('AUTH_SECRET is not set')
  }
  const exp = Math.floor(Date.now() / 1000) + SESSION_TTL_SECONDS
  const token = await signSession({ uid: user.id, email: user.email, exp }, secret)
  response.cookies.set(COOKIE_NAME, token, sessionCookieOptions())
  return response
}
