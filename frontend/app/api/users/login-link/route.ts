import { NextRequest, NextResponse } from 'next/server'
import { membershipFor } from '@/lib/auth/membership'
import { requireAdmin } from '@/lib/auth/member'
import { sendPasswordEmail } from '@/lib/auth/password-reset'
import { serviceSupabase } from '@/lib/auth/supabase-server'

export const dynamic = 'force-dynamic'

export async function POST(request: NextRequest) {
  try {
    const auth = await requireAdmin()
    if (auth.error || !auth.member) return auth.error
    const body = await request.json()
    const userId = typeof body.userId === 'string' ? body.userId : ''
    const membership = userId ? await membershipFor(userId) : null
    if (!membership) {
      return NextResponse.json({ error: 'That person is not a LayerPainter user' }, { status: 404 })
    }
    const { data, error } = await serviceSupabase().auth.admin.getUserById(userId)
    const email = data.user?.email?.toLowerCase() || ''
    if (error || !email) {
      return NextResponse.json({ error: 'Could not find that login' }, { status: 404 })
    }
    const mailed = await sendPasswordEmail(request, email, 'welcome')
    if (mailed.status === 'failed') {
      return NextResponse.json({ error: mailed.error }, { status: mailed.httpStatus })
    }
    if (mailed.status === 'missing') {
      return NextResponse.json({ error: 'Could not send the login email' }, { status: 404 })
    }
    return NextResponse.json({ ok: true })
  } catch (error) {
    console.error('Could not send the login email:', error instanceof Error ? error.message : 'unknown')
    return NextResponse.json({ error: 'Could not send the login email' }, { status: 503 })
  }
}
