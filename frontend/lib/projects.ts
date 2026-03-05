/**
 * Project persistence.
 * Primary store: server-side JSON files via /api/projects.
 * Local cache: localStorage (for quick reads and offline fallback).
 */
import { API_BASE_URL } from '@/lib/config'

const STORAGE_KEY = 'layerpainter_projects'
const MAX_PROJECTS = 100
let syncingPromise: Promise<Project[]> | null = null

export interface Project {
  sessionId: string
  name: string
  imageFileName: string
  libraryGroup: string
  canvasWidthCm: number
  canvasHeightCm: number
  saturationBoost: number
  detailLevel: number
  createdAt: number
  /** Used by Regenerate Layers on Projection tab */
  nColors?: number
  overpaintMm?: number
  orderMode?: string
  maxSide?: number
}

function getStored(): Project[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw) as unknown
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function setStored(projects: Project[]): void {
  if (typeof window === 'undefined') return
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(projects))
  } catch (e) {
    console.error('Failed to save projects:', e)
  }
}

function normalizeProjects(projects: Project[]): Project[] {
  return [...projects]
    .filter((p) => !!p && typeof p.sessionId === 'string' && p.sessionId.length > 0)
    .sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0))
    .slice(0, MAX_PROJECTS)
}

export async function syncProjectsFromServer(force: boolean = false): Promise<Project[]> {
  if (typeof window === 'undefined') return []
  if (!force && syncingPromise) return syncingPromise

  syncingPromise = (async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/projects`, { cache: 'no-store' })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json() as { projects?: Project[] }
      const normalized = normalizeProjects(Array.isArray(data.projects) ? data.projects : [])
      setStored(normalized)
      return normalized
    } catch (e) {
      // Fall back to local cache if server unavailable.
      return normalizeProjects(getStored())
    } finally {
      syncingPromise = null
    }
  })()

  return syncingPromise
}

async function upsertProjectToServer(project: Project): Promise<void> {
  try {
    await fetch(`${API_BASE_URL}/api/projects/${encodeURIComponent(project.sessionId)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(project),
    })
  } catch (e) {
    // Keep local cache even if server sync fails; next sync can reconcile.
  }
}

async function deleteProjectFromServer(sessionId: string): Promise<void> {
  try {
    await fetch(`${API_BASE_URL}/api/projects/${encodeURIComponent(sessionId)}`, {
      method: 'DELETE',
    })
  } catch (e) {
    // Ignore network failures; local cache already updated.
  }
}

/** Returns recent projects, newest first. */
export function getProjects(): Project[] {
  return normalizeProjects(getStored())
}

/** Get a single project by session ID, or undefined. */
export function getProjectBySessionId(sessionId: string): Project | undefined {
  return getStored().find((p) => p.sessionId === sessionId)
}

/** Add or update a project by sessionId. Keeps list bounded. */
export function saveProject(project: Project): void {
  const list = getStored()
  const idx = list.findIndex((p) => p.sessionId === project.sessionId)
  if (idx >= 0) list.splice(idx, 1)
  list.unshift(project)
  const trimmed = normalizeProjects(list)
  setStored(trimmed)
  void upsertProjectToServer(project)
}

/** Remove a project by sessionId. */
export function removeProject(sessionId: string): void {
  const list = getStored().filter((p) => p.sessionId !== sessionId)
  setStored(list)
  void deleteProjectFromServer(sessionId)
}
