import { NextRequest, NextResponse } from 'next/server'
import { membershipFor } from '@/lib/auth/membership'
import { withSessionCookie } from '@/lib/auth/member'
import { routeSupabase } from '@/lib/auth/supabase-server'

export const dynamic = 'force-dynamic'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const password = typeof body.password === 'string' ? body.password : ''
    if (password.length < 6 || password.length > 1024) {
      return NextResponse.json({ error: 'Password must be at least 6 characters.' }, { status: 400 })
    }

    const supabase = routeSupabase()
    const { data: userData, error: userError } = await supabase.auth.getUser()
    if (userError || !userData.user?.email) {
      return NextResponse.json(
        { error: 'Open the reset link from your email again, then choose a new password.' },
        { status: 401 },
      )
    }

    const membership = await membershipFor(userData.user.id).catch(() => null)
    if (!membership) {
      await supabase.auth.signOut()
      return NextResponse.json({ error: 'This email does not have a LayerPainter account.' }, { status: 403 })
    }

    const { error } = await supabase.auth.updateUser({ password })
    if (error) {
      console.error('Password update failed:', error.message)
      return NextResponse.json(
        { error: 'Could not update the password. Request a new reset link and try again.' },
        { status: 400 },
      )
    }

    const response = NextResponse.json({ ok: true })
    await withSessionCookie(response, { id: userData.user.id, email: userData.user.email })
    return response
  } catch (error) {
    console.error('Password update failed:', error instanceof Error ? error.message : 'unknown')
    return NextResponse.json(
      { error: 'Could not update the password. Request a new reset link and try again.' },
      { status: 400 },
    )
  }
}
