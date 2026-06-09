/**
 * Remember position/size of the named popup (window.open(..., 'projection', ...))
 * so it reopens on the same monitor after move/resize.
 */

export const PROJECTION_POPUP_STORAGE_KEY = 'layerpainter_projection_window_bounds'

export interface ProjectionPopupBounds {
  left: number
  top: number
  width: number
  height: number
}

const DEFAULT_WIDTH = 1920
const DEFAULT_HEIGHT = 1080

export function readProjectionPopupBounds(): ProjectionPopupBounds | null {
  if (typeof window === 'undefined') return null
  try {
    const raw = localStorage.getItem(PROJECTION_POPUP_STORAGE_KEY)
    if (!raw) return null
    const b = JSON.parse(raw) as Partial<ProjectionPopupBounds>
    if (
      typeof b.left !== 'number' ||
      typeof b.top !== 'number' ||
      typeof b.width !== 'number' ||
      typeof b.height !== 'number'
    )
      return null
    if (!Number.isFinite(b.width) || !Number.isFinite(b.height) || b.width < 200 || b.height < 200)
      return null
    return {
      left: Math.round(b.left),
      top: Math.round(b.top),
      width: Math.round(b.width),
      height: Math.round(b.height),
    }
  } catch {
    return null
  }
}

/** Call from the projection viewer while it is open (move/resize/close). */
export function writeProjectionPopupBounds(): void {
  if (typeof window === 'undefined') return
  try {
    const left = window.screenX ?? window.screenLeft ?? 0
    const top = window.screenY ?? window.screenTop ?? 0
    const width = window.outerWidth
    const height = window.outerHeight
    if (width < 200 || height < 200) return
    const b: ProjectionPopupBounds = {
      left: Math.round(left),
      top: Math.round(top),
      width: Math.round(width),
      height: Math.round(height),
    }
    localStorage.setItem(PROJECTION_POPUP_STORAGE_KEY, JSON.stringify(b))
  } catch {
    /* ignore quota / private mode */
  }
}

/** Feature string for window.open(..., 'projection', features). */
export function projectionPopupOpenFeatures(): string {
  const b = readProjectionPopupBounds()
  const w = b?.width ?? DEFAULT_WIDTH
  const h = b?.height ?? DEFAULT_HEIGHT
  const base = `width=${w},height=${h},menubar=no,toolbar=no,location=no,status=no`
  if (b && Number.isFinite(b.left) && Number.isFinite(b.top)) {
    return `left=${b.left},top=${b.top},${base}`
  }
  return base
}
