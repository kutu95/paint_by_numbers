import { randomBytes } from 'crypto'
import { NextRequest, NextResponse } from 'next/server'
import { listLayerPainterUsers } from '@/lib/auth/directory'
import {
  deleteMembership,
  insertMembership,
  membershipFor,
  otherAdminCount,
  updateMembershipRole,
} from '@/lib/auth/membership'
import { requireAdmin } from '@/lib/auth/member'
import { findAuthUserByEmail, isPlausibleEmail, sendPasswordEmail } from '@/lib/auth/password-reset'
import { serviceSupabase } from '@/lib/auth/supabase-server'

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    const auth = await requireAdmin()
    if (auth.error || !auth.member) return auth.error
    return NextResponse.json({ users: await listLayerPainterUsers() })
  } catch (error) {
    console.error('Could not list users:', error instanceof Error ? error.message : 'unknown')
    return NextResponse.json({ error: 'Could not list users' }, { status: 500 })
  }
}

export async function POST(request: NextRequest) {
  try {
    const auth = await requireAdmin()
    if (auth.error || !auth.member) return auth.error
    const body = await request.json()
    const email = typeof body.email === 'string' ? body.email.trim().toLowerCase() : ''
    const role = body.role === 'admin' ? 'admin' : body.role === 'user' ? 'user' : ''
    const displayName = typeof body.displayName === 'string' ? body.displayName.trim().slice(0, 80) : ''
    const sendEmail = Boolean(body.sendEmail)
    if (!isPlausibleEmail(email) || !role) {
      return NextResponse.json({ error: 'An email and a role are required' }, { status: 400 })
    }

    let account = await findAuthUserByEmail(email)
    if (!account) {
      const { data, error } = await serviceSupabase().auth.admin.createUser({
        email,
        password: randomBytes(32).toString('base64url'),
        email_confirm: true,
        user_metadata: { app: 'layerpainter', full_name: displayName },
      })
      if (error || !data.user) {
        console.error('Could not create the login:', error?.message || 'missing user')
        return NextResponse.json({ error: 'Could not create that account' }, { status: 503 })
      }
      account = data.user
    }

    const existing = await membershipFor(account.id)
    if (existing) {
      return NextResponse.json({ error: 'That email already has a LayerPainter account' }, { status: 409 })
    }
    const inserted = await insertMembership(account.id, role, displayName || null)
    if (!inserted) {
      return NextResponse.json({ error: 'That email already has a LayerPainter account' }, { status: 409 })
    }

    if (sendEmail) {
      const mailed = await sendPasswordEmail(request, email, 'welcome')
      if (mailed.status === 'failed') {
        return NextResponse.json(
          { error: 'The account was created, but the login email could not be sent.', userId: account.id },
          { status: 503 },
        )
      }
    }
    return NextResponse.json({ ok: true, userId: account.id })
  } catch (error) {
    console.error('Could not create the user:', error instanceof Error ? error.message : 'unknown')
    return NextResponse.json({ error: 'Could not create that account' }, { status: 500 })
  }
}

export async function PATCH(request: NextRequest) {
  try {
    const auth = await requireAdmin()
    if (auth.error || !auth.member) return auth.error
    const body = await request.json()
    const userId = typeof body.userId === 'string' ? body.userId : ''
    const role = body.role === 'admin' ? 'admin' : body.role === 'user' ? 'user' : ''
    if (!userId || !role) {
      return NextResponse.json({ error: 'A user and a valid role are required' }, { status: 400 })
    }
    const existing = await membershipFor(userId)
    if (!existing) {
      return NextResponse.json({ error: 'That person is not a LayerPainter user' }, { status: 404 })
    }
    if (existing.role === 'admin' && role !== 'admin' && (await otherAdminCount(userId)) === 0) {
      return NextResponse.json({ error: 'At least one admin is required' }, { status: 400 })
    }
    await updateMembershipRole(userId, role)
    return NextResponse.json({ ok: true })
  } catch (error) {
    console.error('Could not update the role:', error instanceof Error ? error.message : 'unknown')
    return NextResponse.json({ error: 'Could not update the role' }, { status: 500 })
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const auth = await requireAdmin()
    if (auth.error || !auth.member) return auth.error
    const body = await request.json()
    const userId = typeof body.userId === 'string' ? body.userId : ''
    if (!userId) {
      return NextResponse.json({ error: 'A user is required' }, { status: 400 })
    }
    if (userId === auth.member.id) {
      return NextResponse.json({ error: 'You cannot remove your own access' }, { status: 400 })
    }
    const existing = await membershipFor(userId)
    if (!existing) {
      return NextResponse.json({ error: 'That person is not a LayerPainter user' }, { status: 404 })
    }
    if (existing.role === 'admin' && (await otherAdminCount(userId)) === 0) {
      return NextResponse.json({ error: 'At least one admin is required' }, { status: 400 })
    }
    await deleteMembership(userId)
    return NextResponse.json({ ok: true })
  } catch (error) {
    console.error('Could not remove the user:', error instanceof Error ? error.message : 'unknown')
    return NextResponse.json({ error: 'Could not remove that account' }, { status: 500 })
  }
}
