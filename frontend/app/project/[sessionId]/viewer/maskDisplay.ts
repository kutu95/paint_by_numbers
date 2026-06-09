/**
 * Projection mask display modes (K key cycles white → color → detail).
 */

export type MaskDisplayMode = 'white' | 'color' | 'detail'

export const MASK_DISPLAY_MODES: MaskDisplayMode[] = ['white', 'color', 'detail']

export function cycleMaskDisplayMode(current: MaskDisplayMode): MaskDisplayMode {
  const idx = MASK_DISPLAY_MODES.indexOf(current)
  const next = idx < 0 ? 0 : (idx + 1) % MASK_DISPLAY_MODES.length
  return MASK_DISPLAY_MODES[next]!
}

export function maskDisplayModeLabel(mode: MaskDisplayMode): string {
  switch (mode) {
    case 'white':
      return 'White on black'
    case 'color':
      return 'Color on black'
    case 'detail':
      return 'Detail (white / grey / black)'
    default:
      return mode
  }
}

export function resolveMaskDisplayMode(
  hud: Partial<{ maskDisplayMode?: MaskDisplayMode; showColor?: boolean }>
): MaskDisplayMode {
  if (hud.maskDisplayMode && MASK_DISPLAY_MODES.includes(hud.maskDisplayMode)) {
    return hud.maskDisplayMode
  }
  return hud.showColor ? 'color' : 'white'
}

function isMaskPixel(data: Uint8ClampedArray, i: number): boolean {
  return data[i]! > 0 || data[i + 1]! > 0 || data[i + 2]! > 0
}

function loadMaskImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error(`Failed to load mask: ${url}`))
    img.src = url
  })
}

/** White = final color region; grey = registration / later overpaint; black = skip. */
export async function buildDetailMaskDataUrl(
  expandedMaskUrl: string,
  pureMaskUrl: string
): Promise<string> {
  const [expandedImg, pureImg] = await Promise.all([
    loadMaskImage(expandedMaskUrl),
    loadMaskImage(pureMaskUrl),
  ])
  const w = expandedImg.width
  const h = expandedImg.height
  if (pureImg.width !== w || pureImg.height !== h) {
    throw new Error('Expanded and pure masks have different dimensions')
  }

  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Canvas not available')

  ctx.drawImage(expandedImg, 0, 0)
  const expandedData = ctx.getImageData(0, 0, w, h)

  ctx.drawImage(pureImg, 0, 0)
  const pureData = ctx.getImageData(0, 0, w, h).data

  const out = expandedData
  const data = out.data
  const grey = 140

  for (let i = 0; i < data.length; i += 4) {
    if (isMaskPixel(pureData, i)) {
      data[i] = 255
      data[i + 1] = 255
      data[i + 2] = 255
      data[i + 3] = 255
    } else if (isMaskPixel(data, i)) {
      data[i] = grey
      data[i + 1] = grey
      data[i + 2] = grey
      data[i + 3] = 255
    } else {
      data[i] = 0
      data[i + 1] = 0
      data[i + 2] = 0
      data[i + 3] = 255
    }
  }

  ctx.putImageData(out, 0, 0)
  return canvas.toDataURL()
}

export function buildColorMaskDataUrl(image: HTMLImageElement, colorHex: string): string {
  const canvas = document.createElement('canvas')
  canvas.width = image.width
  canvas.height = image.height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Canvas not available')
  ctx.imageSmoothingEnabled = false
  ctx.drawImage(image, 0, 0)
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
  const data = imageData.data
  const r = parseInt(colorHex.slice(1, 3), 16)
  const g = parseInt(colorHex.slice(3, 5), 16)
  const b = parseInt(colorHex.slice(5, 7), 16)
  for (let i = 0; i < data.length; i += 4) {
    if (isMaskPixel(data, i)) {
      data[i] = r
      data[i + 1] = g
      data[i + 2] = b
      data[i + 3] = 255
    } else {
      data[i] = 0
      data[i + 1] = 0
      data[i + 2] = 0
      data[i + 3] = 255
    }
  }
  ctx.putImageData(imageData, 0, 0)
  return canvas.toDataURL()
}
