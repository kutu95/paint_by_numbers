import { NextRequest, NextResponse } from 'next/server'
import { membershipFor } from '@/lib/auth/membership'
import { routeSupabase } from '@/lib/auth/supabase-server'

export const dynamic = 'force-dynamic'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const tokenHash = typeof body.token_hash === 'string' ? body.token_hash : ''
    if (!tokenHash || tokenHash.length > 2048) {
      return NextResponse.json({ error: 'This reset link is invalid or has expired.' }, { status: 400 })
    }

    const supabase = routeSupabase()
    const { error } = await supabase.auth.verifyOtp({
      token_hash: tokenHash,
      type: 'recovery',
    })
    if (error) {
      console.error('Recovery link verification failed:', error.message)
      return NextResponse.json({ error: 'This reset link is invalid or has expired.' }, { status: 400 })
    }
    const { data } = await supabase.auth.getUser()
    const membership = data.user ? await membershipFor(data.user.id).catch(() => null) : null
    if (!membership) {
      await supabase.auth.signOut()
      return NextResponse.json({ error: 'This email does not have a LayerPainter account.' }, { status: 403 })
    }
    return NextResponse.json({ ok: true })
  } catch (error) {
    console.error('Recovery confirmation failed:', error instanceof Error ? error.message : 'unknown')
    return NextResponse.json({ error: 'This reset link is invalid or has expired.' }, { status: 400 })
  }
}
