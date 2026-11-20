import { NextRequest, NextResponse } from 'next/server'
import { listLayerPainterUsers } from '@/lib/auth/directory'
import { membershipFor } from '@/lib/auth/membership'
import { requireMember } from '@/lib/auth/member'
import { isPlausibleEmail, sendPasswordEmail } from '@/lib/auth/password-reset'
import { provisionMember } from '@/lib/auth/provision-member'
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

async function ownedProject(request: NextRequest, projectId: string, ownerId: string) {
  const { response, payload } = await forward(request, '/api/projects', 'GET')
  if (!response.ok) {
    return { error: NextResponse.json({ error: payload.detail || 'Could not load the project' }, { status: response.status }) }
  }
  const projects = Array.isArray(payload.projects) ? payload.projects : []
  const project = projects.find((item: { sessionId?: string; projectId?: string }) => (
    item.sessionId === projectId || item.projectId === projectId
  ))
  if (!project) {
    return { error: NextResponse.json({ error: 'Project not found' }, { status: 404 }) }
  }
  if (String(project.ownerId || '') !== ownerId) {
    return { error: NextResponse.json({ error: 'Only the project owner can do that' }, { status: 403 }) }
  }
  const name = typeof project.name === 'string' ? project.name.trim() : ''
  return { projectName: name || 'a LayerPainter project' }
}

export async function GET(request: NextRequest, { params }: { params: { projectId: string } }) {
  const auth = await requireMember()
  if (auth.error || !auth.member) return auth.error
  const { response, payload } = await forward(request, `/api/projects/${params.projectId}/collaborators`, 'GET')
  if (!response.ok) {
    return NextResponse.json({ error: payload.detail || 'Could not load collaborators' }, { status: response.status })
  }
  const ids = Array.isArray(payload.collaboratorIds) ? payload.collaboratorIds.map(String) : []
  const owned = await ownedProject(request, params.projectId, auth.member.id)
  const directory = owned.error
    ? []
    : (await listLayerPainterUsers())
      .filter((user) => user.userId !== auth.member.id && user.email)
      .map((user) => ({
        userId: user.userId,
        email: user.email,
        displayName: user.displayName,
      }))
  return NextResponse.json({ collaborators: await withEmails(ids), directory })
}

export async function POST(request: NextRequest, { params }: { params: { projectId: string } }) {
  const auth = await requireMember()
  if (auth.error || !auth.member) return auth.error
  const owned = await ownedProject(request, params.projectId, auth.member.id)
  if ('error' in owned) return owned.error
  const body = await request.json().catch(() => ({}))
  const requestedId = typeof body.userId === 'string' ? body.userId.trim() : ''
  const email = typeof body.email === 'string' ? body.email.trim().toLowerCase() : ''

  let targetId = requestedId
  let created = false
  if (requestedId) {
    const membership = await membershipFor(requestedId)
    if (!membership) {
      return NextResponse.json({ error: 'That person does not have a LayerPainter account.' }, { status: 404 })
    }
  } else if (isPlausibleEmail(email)) {
    try {
      const provisioned = await provisionMember(email)
      targetId = provisioned.userId
      created = provisioned.created
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Could not create that account'
      return NextResponse.json({ error: message }, { status: 503 })
    }
  } else {
    return NextResponse.json({ error: 'Choose a user or enter an email address.' }, { status: 400 })
  }

  if (targetId === auth.member.id) {
    return NextResponse.json({ error: 'You already have this project.' }, { status: 400 })
  }

  let emailWarning: string | null = null
  if (created) {
    const mailed = await sendPasswordEmail(request, email, 'welcome', { projectName: owned.projectName })
    if (mailed.status !== 'sent') {
      emailWarning = 'The account was created, but the invitation email could not be sent.'
    }
  }

  const { response, payload } = await forward(
    request,
    `/api/projects/${params.projectId}/collaborators`,
    'POST',
    { userId: targetId },
  )
  if (!response.ok) {
    return NextResponse.json({ error: payload.detail || 'Could not invite that person' }, { status: response.status })
  }
  const ids = Array.isArray(payload.collaboratorIds) ? payload.collaboratorIds.map(String) : []
  const collaborators = await withEmails(ids)
  if (emailWarning) {
    return NextResponse.json({ error: emailWarning, collaborators, created: true }, { status: 503 })
  }
  return NextResponse.json({ collaborators, created })
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
