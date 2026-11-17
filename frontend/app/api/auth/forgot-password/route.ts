import { NextRequest, NextResponse } from 'next/server'
import { isPlausibleEmail, sendPasswordEmail } from '@/lib/auth/password-reset'

const attempts = new Map<string, number[]>()

function limited(key: string, max: number, windowMs: number) {
  const now = Date.now()
  const recent = (attempts.get(key) || []).filter((time) => now - time < windowMs)
  if (recent.length >= max) {
    attempts.set(key, recent)
    return true
  }
  recent.push(now)
  attempts.set(key, recent)
  return false
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const email = typeof body.email === 'string' ? body.email.trim().toLowerCase() : ''
    if (!isPlausibleEmail(email)) {
      return NextResponse.json({ error: 'Enter a valid email address.' }, { status: 400 })
    }
    const ip = request.headers.get('x-forwarded-for')?.split(',')[0]?.trim() || 'local'
    if (limited(`email:${email}`, 5, 60 * 60 * 1000) || limited(`ip:${ip}`, 20, 60 * 60 * 1000)) {
      return NextResponse.json({ error: 'Too many reset requests. Try again later.' }, { status: 429 })
    }

    const result = await sendPasswordEmail(request, email, 'reset')
    if (result.status === 'missing') return NextResponse.json({ ok: true })
    if (result.status === 'failed') {
      return NextResponse.json({ error: result.error }, { status: result.httpStatus })
    }
    return NextResponse.json({ ok: true })
  } catch (error) {
    console.error('Forgot password error:', error instanceof Error ? error.message : 'unknown')
    return NextResponse.json({ error: 'Could not send the reset email. Try again later.' }, { status: 503 })
  }
}
