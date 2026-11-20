import { NextRequest, NextResponse } from 'next/server'
import { listLayerPainterUsers } from '@/lib/auth/directory'
import { requireMember } from '@/lib/auth/member'

export const dynamic = 'force-dynamic'

type Person = {
  userId: string
  email: string
  displayName: string | null
}

function backendBase() {
  return (process.env.BACKEND_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')
}

export async function GET(request: NextRequest) {
  const auth = await requireMember()
  if (auth.error || !auth.member) return auth.error
  const response = await fetch(`${backendBase()}/api/projects`, {
    headers: { cookie: request.headers.get('cookie') || '' },
    cache: 'no-store',
  })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    return NextResponse.json({ error: payload.detail || 'Could not load projects' }, { status: response.status })
  }
  const projects = Array.isArray(payload.projects) ? payload.projects : []
  if (auth.member.role !== 'admin') {
    return NextResponse.json({ projects })
  }

  try {
    const users = await listLayerPainterUsers()
    const byId = new Map<string, Person>(users.map((user) => [user.userId, {
      userId: user.userId,
      email: user.email,
      displayName: user.displayName,
    }]))
    const person = (userId: unknown): Person | null => {
      if (typeof userId !== 'string' || !userId) return null
      return byId.get(userId) || { userId, email: '', displayName: null }
    }
    const enriched = projects.map((project: { ownerId?: unknown; collaboratorIds?: unknown }) => {
      const collaboratorIds = Array.isArray(project.collaboratorIds) ? project.collaboratorIds : []
      return {
        ...project,
        owner: person(project.ownerId),
        collaborators: collaboratorIds.map((userId) => person(userId)).filter((item): item is Person => item !== null),
      }
    })
    return NextResponse.json({ projects: enriched })
  } catch (error) {
    console.error('Could not name project members:', error instanceof Error ? error.message : 'unknown')
    return NextResponse.json({ projects })
  }
}
