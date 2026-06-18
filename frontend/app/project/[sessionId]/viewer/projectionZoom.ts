export const PROJECTION_SCALE_MIN_PCT = 25
export const PROJECTION_SCALE_MAX_PCT = 200
export const PROJECTION_SCALE_COARSE_STEP_PCT = 5
export const PROJECTION_SCALE_FINE_STEP_PCT = 1
export const PROJECTION_ZOOM_OVERLAY_MS = 3000

export function clampProjectionScalePercent(pct: number): number {
  return Math.max(
    PROJECTION_SCALE_MIN_PCT,
    Math.min(PROJECTION_SCALE_MAX_PCT, Math.round(pct))
  )
}

export function scaleToPercent(scale: number): number {
  return clampProjectionScalePercent(Math.round(scale * 100))
}

export function percentToScale(pct: number): number {
  return clampProjectionScalePercent(pct) / 100
}

export function adjustProjectionScalePercent(currentPct: number, deltaPct: number): number {
  return clampProjectionScalePercent(currentPct + deltaPct)
}

export function formatProjectionScalePercent(scale: number): string {
  return `${scaleToPercent(scale)}%`
}
