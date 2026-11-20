/**
 * Keyboard shortcuts for the projection viewer window.
 * Keep in sync with handleKeyPress in viewer/page.tsx.
 *
 * PROJECTION_SHORTCUTS_LINES is the short reminder on the projection window
 * and the separate HUD window. PROJECTION_SHORTCUT_GROUPS is the full
 * explanation shown on the Projection tab.
 */

/** Multi-line reference (on-screen HUD overlay in projection window). */
export const PROJECTION_SHORTCUTS_LINES: string[] = [
  '← → Space · D done · S show done · H hide HUD',
  'C crosshairs · X grid · K mask · I invert · L pure · O outline',
  '[ ] opacity · − + scale 5% · Shift − + fine 1% · F final · G original · R registration',
  'B black · W white · E end lasso · Enter close lasso · Esc cancel/close',
]

export interface ProjectionShortcut {
  /** Key labels, in the order they should be read. */
  keys: string[]
  /** What the key does in the projection window. */
  detail: string
}

export interface ProjectionShortcutGroup {
  title: string
  shortcuts: ProjectionShortcut[]
}

export const PROJECTION_SHORTCUT_GROUPS: ProjectionShortcutGroup[] = [
  {
    title: 'Layers',
    shortcuts: [
      {
        keys: ['←'],
        detail:
          'Previous layer. Stops at the first layer. While Show done is off, layers marked done are skipped; the finished layer is still reachable. When a lasso is closed and has paint inside it, ← stays inside those layers and wraps from the first back to the last.',
      },
      {
        keys: ['→'],
        detail:
          'Next layer, with the same skipping and lasso rules as ←. Stops at the last layer unless a lasso is closed.',
      },
      {
        keys: ['Space'],
        detail:
          'Next layer, the same step as →. The projection window does not scroll.',
      },
      {
        keys: ['D'],
        detail:
          'Marks the current layer done, or clears that mark. Done layers are skipped by ←, →, and Space while Show done is off. The finished layer cannot be marked done.',
      },
      {
        keys: ['S'],
        detail:
          'Includes layers marked done when you step with ←, →, or Space. Press S again to skip those layers. The picture stays on the current layer until you step.',
      },
    ],
  },
  {
    title: 'Mask',
    shortcuts: [
      {
        keys: ['K'],
        detail:
          'Cycles the mask view: white on black, then this layer’s color on black, then the detail map, then back to white on black. White on black is the mask image. Color on black fills the paint region with the layer’s palette color. Detail is white where this color stays, grey where a later layer will cover the overlap, and black everywhere else. Leaving white on black also turns invert off.',
      },
      {
        keys: ['I'],
        detail:
          'Inverts the projected picture, so white becomes black and colors flip. Turning invert on also switches the mask view to white on black.',
      },
      {
        keys: ['L'],
        detail:
          'Switches the layer mask between expanded and pure. Expanded includes the overlap the next layer registers against. Pure is only the region that keeps this color. White-on-black and color views follow this choice, and so does an active lasso when it looks for paint inside the loop. The detail view always uses both masks.',
      },
      {
        keys: ['O'],
        detail:
          'Cycles the outline on the layer mask: off, thin, thick, glow, then off again. The outline is drawn with the layer mask. It is not drawn over the final preview or the original image.',
      },
      {
        keys: ['['],
        detail: 'Lowers the layer-mask opacity by 5 points, down to 40%. The final preview and the original image stay fully opaque.',
      },
      {
        keys: [']'],
        detail: 'Raises the layer-mask opacity by 5 points, up to 100%.',
      },
    ],
  },
  {
    title: 'Alignment',
    shortcuts: [
      {
        keys: ['C'],
        detail: 'Shows or hides a white cross through the center of the projection window.',
      },
      {
        keys: ['X'],
        detail: 'Shows or hides a grid of lines spaced every 5% of the window.',
      },
      {
        keys: ['R'],
        detail:
          'Hides the projected picture so the window is black. Crosshairs and the grid stay visible if they are on, so you can line the projector up with the canvas. Press R again to bring the picture back.',
      },
      {
        keys: ['H'],
        detail:
          'Shows or hides the short layer name and key reminder overlaid on the projection window. This Projection tab is unchanged.',
      },
    ],
  },
  {
    title: 'Scale',
    shortcuts: [
      {
        keys: ['−', 'Numpad −'],
        detail:
          'Shrinks the picture by 5%. Numpad − does the same thing. Scale stays between 25% and 200%, and the new percentage flashes in the center of the projection for a few seconds.',
      },
      {
        keys: ['=', 'Numpad +'],
        detail:
          'Enlarges the picture by 5%. Use the = key (the main-keyboard key that also has + printed on it) without Shift. Numpad + does the same thing. Holding Shift on this key is the 1% step, so the 5% enlarge is the unshifted = key.',
      },
      {
        keys: ['Shift −', 'Shift =', 'Shift Numpad −', 'Shift Numpad +'],
        detail:
          'Changes the scale by 1% in the same direction as the key without Shift. Shift with = is the combination that prints +.',
      },
    ],
  },
  {
    title: 'Reference pictures',
    shortcuts: [
      {
        keys: ['F'],
        detail:
          'Shows the finished preview, or returns to the current layer. Turning the final preview on also leaves the original image.',
      },
      {
        keys: ['G'],
        detail:
          'Shows the original uploaded image, or returns to the current layer. Turning the original on also leaves the final preview.',
      },
    ],
  },
  {
    title: 'Screen',
    shortcuts: [
      {
        keys: ['B'],
        detail:
          'Covers the whole projection in black, including the picture, crosshairs, grid, and on-screen HUD. Press B again to restore them. Turns the white screen off.',
      },
      {
        keys: ['W'],
        detail:
          'Covers the whole projection in white, with the same coverage as B. Press W again to restore the picture. Turns the black screen off.',
      },
    ],
  },
  {
    title: 'Lasso',
    shortcuts: [
      {
        keys: ['Enter'],
        detail:
          'While a lasso is being drawn, Enter closes the shape once it has at least three points. The projection is then clipped to that shape, and ←, →, and Space move only through layers that have paint inside it.',
      },
      {
        keys: ['E'],
        detail:
          'Clears a lasso you are drawing or one that is already closed, and returns to the full picture.',
      },
      {
        keys: ['Esc'],
        detail:
          'While a lasso is being drawn, Esc drops the points and leaves lasso mode. A lasso that is already closed stays up; press E to clear it. In every other case, Esc closes the projection window.',
      },
    ],
  },
]
