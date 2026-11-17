import { NextResponse } from 'next/server'
import { COOKIE_NAME, sessionCookieOptions } from '@/lib/auth/session-cookie'

export const dynamic = 'force-dynamic'

export async function POST() {
  const response = NextResponse.json({ ok: true })
  response.cookies.set(COOKIE_NAME, '', { ...sessionCookieOptions(0), maxAge: 0 })
  return response
}
