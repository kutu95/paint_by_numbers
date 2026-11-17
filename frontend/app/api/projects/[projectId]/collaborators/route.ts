import { NextRequest, NextResponse } from 'next/server'
import { membershipFor } from '@/lib/auth/membership'
import { requireMember } from '@/lib/auth/member'
import { findAuthUserByEmail, isPlausibleEmail } from '@/lib/auth/password-reset'
import { serviceSupabase } from '@/lib/auth/supabase-server'

export const dynamic = 'force-dynamic'

function backendBase() {
  return (process.env.BACKEND_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')
}

async function forward(request: NextRequest, path: string, method: string, body?: unknown) {
  const response = await fetch(`${backendBase()}${path}`, {
    method,
    headers: {
      cookie: request.headers.get('cookie') || '',
      'content-type': 'application/json',
    },
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: 'no-store',
  })
  const payload = await response.json().catch(() => ({}))
  return { response, payload }
}

async function withEmails(ids: string[]) {
  const supabase = serviceSupabase()
  const collaborators = []
  for (const userId of ids) {
    const { data } = await supabase.auth.admin.getUserById(userId)
    const membership = await membershipFor(userId)
    collaborators.push({
      userId,
      email: data.user?.email || '',
      displayName: membership?.displayName || null,
    })
  }
  return collaborators
}

export async function GET(request: NextRequest, { params }: { params: { projectId: string } }) {
  const auth = await requireMember()
  if (auth.error || !auth.member) return auth.error
  const { response, payload } = await forward(request, `/api/projects/${params.projectId}/collaborators`, 'GET')
  if (!response.ok) {
    return NextResponse.json({ error: payload.detail || 'Could not load collaborators' }, { status: response.status })
  }
  const ids = Array.isArray(payload.collaboratorIds) ? payload.collaboratorIds.map(String) : []
  return NextResponse.json({ collaborators: await withEmails(ids) })
}

export async function POST(request: NextRequest, { params }: { params: { projectId: string } }) {
  const auth = await requireMember()
  if (auth.error || !auth.member) return auth.error
  const body = await request.json().catch(() => ({}))
  const email = typeof body.email === 'string' ? body.email.trim().toLowerCase() : ''
  if (!isPlausibleEmail(email)) {
    return NextResponse.json({ error: 'Enter a valid email address.' }, { status: 400 })
  }
  const account = await findAuthUserByEmail(email)
  const membership = account ? await membershipFor(account.id) : null
  if (!account || !membership) {
    return NextResponse.json({ error: 'That email does not have a LayerPainter account.' }, { status: 404 })
  }
  if (account.id === auth.member.id) {
    return NextResponse.json({ error: 'You already have this project.' }, { status: 400 })
  }
  const { response, payload } = await forward(
    request,
    `/api/projects/${params.projectId}/collaborators`,
    'POST',
    { userId: account.id },
  )
  if (!response.ok) {
    return NextResponse.json({ error: payload.detail || 'Could not invite that person' }, { status: response.status })
  }
  const ids = Array.isArray(payload.collaboratorIds) ? payload.collaboratorIds.map(String) : []
  return NextResponse.json({ collaborators: await withEmails(ids) })
}

export async function DELETE(request: NextRequest, { params }: { params: { projectId: string } }) {
  const auth = await requireMember()
  if (auth.error || !auth.member) return auth.error
  const body = await request.json().catch(() => ({}))
  const userId = typeof body.userId === 'string' ? body.userId : ''
  if (!userId) {
    return NextResponse.json({ error: 'A collaborator is required' }, { status: 400 })
  }
  const { response, payload } = await forward(
    request,
    `/api/projects/${params.projectId}/collaborators/${encodeURIComponent(userId)}`,
    'DELETE',
  )
  if (!response.ok) {
    return NextResponse.json({ error: payload.detail || 'Could not remove that person' }, { status: response.status })
  }
  const ids = Array.isArray(payload.collaboratorIds) ? payload.collaboratorIds.map(String) : []
  return NextResponse.json({ collaborators: await withEmails(ids) })
}
