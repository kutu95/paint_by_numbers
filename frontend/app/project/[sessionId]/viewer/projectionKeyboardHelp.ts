/**
 * Keyboard shortcuts for the projection viewer window.
 * Keep in sync with handleKeyPress in viewer/page.tsx.
 */

/** Single-line reference (Projection tab footer). */
export const PROJECTION_SHORTCUTS_LINE =
  '← → Space: Layer | D: Done | S: Show done | H: HUD | C: Crosshairs | X: Grid | K: Mask mode | I: Invert | L: Pure mask | O: Outline | [ ]: Opacity | − +: Scale | F: Final | G: Original | R: Registration | B/W: Screen | E: End lasso | Enter: Close lasso | Esc: Cancel/Close'

/** Multi-line reference (on-screen HUD overlay in projection window). */
export const PROJECTION_SHORTCUTS_LINES: string[] = [
  '← → Space · D done · S show done · H hide HUD',
  'C crosshairs · X grid · K mask · I invert · L pure · O outline',
  '[ ] opacity · − + scale · F final · G original · R registration',
  'B black · W white · E end lasso · Enter close lasso · Esc cancel/close',
]
