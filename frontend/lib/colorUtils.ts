/**
 * sRGB <-> Lab and ΔE for frontend (e.g. virtual mixer).
 * Uses D65 reference white; simple ΔE = Euclidean distance in Lab.
 */

export function sRgbToLinear(c: number): number {
  const n = c / 255
  return n <= 0.04045 ? n / 12.92 : Math.pow((n + 0.055) / 1.055, 2.4)
}

export function rgbToLab(r: number, g: number, b: number): [number, number, number] {
  const rl = sRgbToLinear(r)
  const gl = sRgbToLinear(g)
  const bl = sRgbToLinear(b)
  // sRGB D65 -> XYZ (matrix)
  let x = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
  let y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
  let z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041
  const xn = 0.95047, yn = 1, zn = 1.08883
  x /= xn
  y /= yn
  z /= zn
  const f = (t: number) => (t > 0.00885645 ? Math.pow(t, 1 / 3) : (7.787037 * t) + 16 / 116)
  const L = 116 * f(y) - 16
  const a = 500 * (f(x) - f(y))
  const bLab = 200 * (f(y) - f(z))
  return [L, a, bLab]
}

export function hexToLab(hex: string): [number, number, number] | null {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
  if (!m) return null
  const r = parseInt(m[1], 16)
  const g = parseInt(m[2], 16)
  const b = parseInt(m[3], 16)
  return rgbToLab(r, g, b)
}

/** ΔE (Euclidean in Lab). */
export function deltaE(lab1: [number, number, number], lab2: [number, number, number]): number {
  return Math.sqrt(
    (lab1[0] - lab2[0]) ** 2 + (lab1[1] - lab2[1]) ** 2 + (lab1[2] - lab2[2]) ** 2
  )
}
