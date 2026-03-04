/**
 * Project persistence in localStorage.
 * Each project has: name, sessionId, original image file name, paint library, canvas size, vibrancy, detail level.
 */

const STORAGE_KEY = 'layerpainter_projects'
const MAX_PROJECTS = 100

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

/** Returns recent projects, newest first. */
export function getProjects(): Project[] {
  const list = getStored()
  return [...list].sort((a, b) => b.createdAt - a.createdAt)
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
  const trimmed = list.slice(0, MAX_PROJECTS)
  setStored(trimmed)
}

/** Remove a project by sessionId. */
export function removeProject(sessionId: string): void {
  const list = getStored().filter((p) => p.sessionId !== sessionId)
  setStored(list)
}
