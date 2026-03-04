/**
 * Lasso tool state shared between Projection tab (tools panel) and projection viewer.
 * Mode: '' = off, 'drawing' = user is drawing lasso in viewer, 'active' = lasso applied, mask clipped.
 */

export type LassoMode = '' | 'drawing' | 'active'

export interface LassoPoint {
  x: number
  y: number
}

const LASSO_MODE_KEY = (id: string) => `projection_lasso_mode_${id}`
const LASSO_PATH_KEY = (id: string) => `projection_lasso_path_${id}`

export function getLassoModeKey(sessionId: string): string {
  return LASSO_MODE_KEY(sessionId)
}

export function getLassoPathKey(sessionId: string): string {
  return LASSO_PATH_KEY(sessionId)
}

export function getLassoMode(sessionId: string): LassoMode {
  if (typeof window === 'undefined') return ''
  const raw = localStorage.getItem(LASSO_MODE_KEY(sessionId))
  if (raw === 'drawing' || raw === 'active') return raw
  return ''
}

export function setLassoMode(sessionId: string, mode: LassoMode): void {
  if (typeof window === 'undefined') return
  if (mode === '') {
    localStorage.removeItem(LASSO_MODE_KEY(sessionId))
    localStorage.removeItem(LASSO_PATH_KEY(sessionId))
  } else {
    localStorage.setItem(LASSO_MODE_KEY(sessionId), mode)
  }
}

export function getLassoPath(sessionId: string): LassoPoint[] | null {
  if (typeof window === 'undefined') return null
  try {
    const raw = localStorage.getItem(LASSO_PATH_KEY(sessionId))
    if (!raw) return null
    const parsed = JSON.parse(raw) as unknown
    if (!Array.isArray(parsed)) return null
    return parsed.filter((p): p is LassoPoint => typeof p === 'object' && p !== null && typeof (p as LassoPoint).x === 'number' && typeof (p as LassoPoint).y === 'number')
  } catch {
    return null
  }
}

export function setLassoPath(sessionId: string, path: LassoPoint[]): void {
  if (typeof window === 'undefined') return
  localStorage.setItem(LASSO_PATH_KEY(sessionId), JSON.stringify(path))
}
