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
  favorSkinTones?: boolean
  skinToneStrength?: number
  easyPainting?: boolean
  easySimplify?: number
  easyFaceDetail?: boolean
  stylePreset?: string
  detailEyes?: boolean
  detailFace?: boolean
  detailBodyOutline?: boolean
  priorityRegionStrength?: number
  hasPriorityRegion?: boolean
  mustIncludeColors?: string[]
  createdAt: number
  /** Used by Regenerate Layers on Projection tab */
  nColors?: number
  overpaintMm?: number
  orderMode?: string
  maxSide?: number
  /** Server list endpoint — oriented source thumbnail URL */
  thumbUrl?: string | null
  hasArtifacts?: boolean
  hasSource?: boolean
  updatedAt?: number
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
      const localBefore = getStored()
      const normalized = normalizeProjects(Array.isArray(data.projects) ? data.projects : [])
      const merged = normalized.map((serverProj) => {
        const localProj = localBefore.find((p) => p.sessionId === serverProj.sessionId)
        return mergeServerProjectWithLocal(serverProj, localProj)
      })
      setStored(merged)
      return merged
    } catch (e) {
      // Fall back to local cache if server unavailable.
      return normalizeProjects(getStored())
    } finally {
      syncingPromise = null
    }
  })()

  return syncingPromise
}

const IMAGE_TAB_MERGE_KEYS: (keyof Project)[] = [
  'favorSkinTones',
  'skinToneStrength',
  'mustIncludeColors',
  'priorityRegionStrength',
]

function mergeServerProjectWithLocal(server: Project, local?: Project): Project {
  if (!local || local.sessionId !== server.sessionId) return server
  const merged: Project = { ...server }
  for (const key of IMAGE_TAB_MERGE_KEYS) {
    const localVal = local[key]
    if (localVal === undefined) continue
    // Stale empty local cache must not wipe server must-include picks after reload.
    if (key === 'mustIncludeColors') {
      const localColors = Array.isArray(localVal) ? localVal : []
      const serverColors = Array.isArray(server.mustIncludeColors) ? server.mustIncludeColors : []
      if (localColors.length === 0 && serverColors.length > 0) continue
    }
    if (JSON.stringify(localVal) !== JSON.stringify(server[key])) {
      Object.assign(merged, { [key]: localVal })
    }
  }
  return merged
}

async function upsertProjectToServer(project: Project): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/projects/${encodeURIComponent(project.sessionId)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(project),
  })
  if (!res.ok) {
    throw new Error(`Project save failed: HTTP ${res.status}`)
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

/** Paint library for API calls — chosen on the Projection tab and stored on the project. */
export function resolveProjectLibraryGroup(sessionId?: string | null): string {
  if (!sessionId) return 'default'
  return getProjectBySessionId(sessionId)?.libraryGroup || 'default'
}

export interface SaveProjectOptions {
  /** Wait for server manifest write (use for Image-tab settings that must survive tab switches). */
  awaitServer?: boolean
}

/** Add or update a project by sessionId. Keeps list bounded. */
export async function saveProject(project: Project, options?: SaveProjectOptions): Promise<void> {
  const list = getStored()
  const idx = list.findIndex((p) => p.sessionId === project.sessionId)
  if (idx >= 0) list.splice(idx, 1)
  list.unshift(project)
  const trimmed = normalizeProjects(list)
  setStored(trimmed)
  if (options?.awaitServer) {
    try {
      await upsertProjectToServer(project)
    } catch (e) {
      console.error('Failed to save project to server:', e)
    }
  } else {
    void upsertProjectToServer(project).catch((e) => {
      console.error('Failed to save project to server:', e)
    })
  }
}

/** Remove a project by sessionId. */
export function removeProject(sessionId: string): void {
  const list = getStored().filter((p) => p.sessionId !== sessionId)
  setStored(list)
  void deleteProjectFromServer(sessionId)
}
