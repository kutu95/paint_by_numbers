/**
 * Shared state between the projection viewer window and the Projection tab HUD controls.
 * Stored in localStorage so both can read/write; the viewer listens for storage events.
 */

export type OutlineMode = 'off' | 'thin' | 'thick' | 'glow'

export type MaskDisplayMode = 'white' | 'color' | 'detail'

export interface ProjectionViewerHudState {
  maskOpacity: number
  usePureMask: boolean
  outlineMode: OutlineMode
  crosshairs: boolean
  grid: boolean
  registrationMode: boolean
  showDoneLayers: boolean
  showFinalPreview: boolean
  showOriginalImage: boolean
  inverted: boolean
  /** K cycles: white on black → palette color → white/grey/black detail map. */
  maskDisplayMode: MaskDisplayMode
  /** @deprecated Use maskDisplayMode === 'color' */
  showColor?: boolean
  blackScreen: boolean
  whiteScreen: boolean
  /** Layer / palette readout overlay on the projection window (toggle with H). */
  showHudOverlay: boolean
}

export const DEFAULT_HUD_STATE: ProjectionViewerHudState = {
  maskOpacity: 85,
  usePureMask: false,
  outlineMode: 'off',
  crosshairs: true,
  grid: false,
  registrationMode: false,
  showDoneLayers: false,
  showFinalPreview: false,
  showOriginalImage: false,
  inverted: false,
  maskDisplayMode: 'white',
  blackScreen: false,
  whiteScreen: false,
  showHudOverlay: false,
}

const KEY_PREFIX = 'projection_viewer_hud_'

export function getViewerHudKey(sessionId: string): string {
  return `${KEY_PREFIX}${sessionId}`
}

export function loadViewerHudState(sessionId: string): ProjectionViewerHudState {
  if (typeof window === 'undefined') return DEFAULT_HUD_STATE
  try {
    const raw = localStorage.getItem(getViewerHudKey(sessionId))
    if (!raw) return DEFAULT_HUD_STATE
    const parsed = JSON.parse(raw) as Partial<ProjectionViewerHudState>
    const merged = { ...DEFAULT_HUD_STATE, ...parsed }
    if (!parsed.maskDisplayMode && parsed.showColor === true) {
      merged.maskDisplayMode = 'color'
    }
    return merged
  } catch {
    return DEFAULT_HUD_STATE
  }
}

export function saveViewerHudState(sessionId: string, state: Partial<ProjectionViewerHudState>): void {
  if (typeof window === 'undefined') return
  try {
    const current = loadViewerHudState(sessionId)
    const next = { ...current, ...state }
    localStorage.setItem(getViewerHudKey(sessionId), JSON.stringify(next))
  } catch (e) {
    console.error('Failed to save viewer HUD state', e)
  }
}
