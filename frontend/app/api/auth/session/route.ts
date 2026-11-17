import { NextResponse } from 'next/server'
import { membershipFor } from '@/lib/auth/membership'
import { withSessionCookie } from '@/lib/auth/member'
import { routeSupabase } from '@/lib/auth/supabase-server'

export const dynamic = 'force-dynamic'

export async function POST() {
  try {
    const supabase = routeSupabase()
    const { data, error } = await supabase.auth.getUser()
    if (error || !data.user?.email) {
      return NextResponse.json({ error: 'Sign in failed. Check the email and password.' }, { status: 401 })
    }
    const membership = await membershipFor(data.user.id)
    if (!membership) {
      return NextResponse.json({ error: 'This email does not have a LayerPainter account.' }, { status: 403 })
    }
    const response = NextResponse.json({
      ok: true,
      email: data.user.email,
      role: membership.role,
    })
    await withSessionCookie(response, { id: data.user.id, email: data.user.email })
    return response
  } catch (error) {
    console.error('Could not start the LayerPainter session:', error instanceof Error ? error.message : 'unknown')
    return NextResponse.json({ error: 'Could not start the session. Try again later.' }, { status: 503 })
  }
}
