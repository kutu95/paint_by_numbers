'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

export type PriorityRegionTool = 'brush' | 'lasso' | 'picker'
export type MaskOverlayMode = 'show' | 'dim' | 'hide'

type Props = {
  imageSrc: string | null
  initialMaskUrl?: string | null
  brushSize: number
  onBrushSizeChange: (size: number) => void
  onMaskChange: (blob: Blob | null) => void
  /** 0–1 strength sent to the pipeline when a region is drawn */
  detailInRegion?: number
  onDetailInRegionChange?: (value: number) => void
  viewportHeightPx?: number
  /** Hex colours (#RRGGBB) forced into the generated palette */
  mustIncludeColors?: string[]
  onMustIncludeColorsChange?: (colors: string[]) => void
  /** Upper bound for must-include picks (typically nColors - 1) */
  maxMustIncludeColors?: number
}

type ColorPickState = {
  hex: string
  clientX: number
  clientY: number
}

const MIN_ZOOM = 1
const MAX_ZOOM = 6
const DEFAULT_VIEWPORT_HEIGHT_PX = 320

const MASK_OVERLAY_OPACITY: Record<MaskOverlayMode, number> = {
  show: 0.45,
  dim: 0.18,
  hide: 0,
}

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n))
}

function normalizeHex(hex: string): string {
  let s = hex.trim().toUpperCase()
  if (!s.startsWith('#')) s = `#${s}`
  return s
}

function rgbToHex(r: number, g: number, b: number): string {
  return `#${[r, g, b].map((x) => x.toString(16).padStart(2, '0')).join('')}`.toUpperCase()
}

function sampleAverageColor(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  radius: number,
  w: number,
  h: number
): string {
  const left = Math.max(0, Math.floor(x - radius))
  const top = Math.max(0, Math.floor(y - radius))
  const rw = Math.min(w - left, radius * 2 + 1)
  const rh = Math.min(h - top, radius * 2 + 1)
  const data = ctx.getImageData(left, top, rw, rh).data
  let sr = 0
  let sg = 0
  let sb = 0
  let n = 0
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] < 16) continue
    sr += data[i]
    sg += data[i + 1]
    sb += data[i + 2]
    n++
  }
  if (n === 0) return '#000000'
  return rgbToHex(Math.round(sr / n), Math.round(sg / n), Math.round(sb / n))
}

function clientToNatural(
  clientX: number,
  clientY: number,
  img: HTMLImageElement
): { x: number; y: number } {
  const rect = img.getBoundingClientRect()
  const x = ((clientX - rect.left) / rect.width) * img.naturalWidth
  const y = ((clientY - rect.top) / rect.height) * img.naturalHeight
  return {
    x: Math.max(0, Math.min(img.naturalWidth - 1, x)),
    y: Math.max(0, Math.min(img.naturalHeight - 1, y)),
  }
}

export function PriorityRegionEditor({
  imageSrc,
  initialMaskUrl,
  brushSize,
  onBrushSizeChange,
  onMaskChange,
  detailInRegion,
  onDetailInRegionChange,
  viewportHeightPx = DEFAULT_VIEWPORT_HEIGHT_PX,
  mustIncludeColors = [],
  onMustIncludeColorsChange,
  maxMustIncludeColors = 15,
}: Props) {
  const imgRef = useRef<HTMLImageElement>(null)
  const maskRef = useRef<HTMLCanvasElement>(null)
  const sampleCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const viewportRef = useRef<HTMLDivElement>(null)
  const [tool, setTool] = useState<PriorityRegionTool>('brush')
  const [pickerHover, setPickerHover] = useState<ColorPickState | null>(null)
  const [lockedPick, setLockedPick] = useState<ColorPickState | null>(null)
  const [drawing, setDrawing] = useState(false)
  const [lassoPoints, setLassoPoints] = useState<{ x: number; y: number }[]>([])
  const lassoRef = useRef<{ x: number; y: number }[]>([])
  const userClearedMaskRef = useRef(false)
  const loadedInitialMaskUrlRef = useRef<string | null>(null)
  const [imageReady, setImageReady] = useState(false)
  const [naturalSize, setNaturalSize] = useState({ w: 0, h: 0 })
  const [fitScale, setFitScale] = useState(1)
  const [zoom, setZoom] = useState(1)
  const [maskOverlayMode, setMaskOverlayMode] = useState<MaskOverlayMode>('show')

  const displayScale = fitScale * zoom
  const displayW = naturalSize.w > 0 ? naturalSize.w * displayScale : 0
  const displayH = naturalSize.h > 0 ? naturalSize.h * displayScale : 0

  const recomputeFitScale = useCallback((nw: number, nh: number) => {
    const vp = viewportRef.current
    if (!vp || nw <= 0 || nh <= 0) return
    const fs = Math.min(vp.clientWidth / nw, viewportHeightPx / nh)
    setFitScale(fs > 0 ? fs : 1)
  }, [viewportHeightPx])

  const applyImageDimensions = useCallback(
    (nw: number, nh: number) => {
      if (nw <= 0 || nh <= 0) return
      setNaturalSize({ w: nw, h: nh })
      setImageReady(true)
      requestAnimationFrame(() => recomputeFitScale(nw, nh))
    },
    [recomputeFitScale]
  )

  // Preload dimensions so we can size the stage before paint (avoids displayW>0 gate blocking mount).
  useEffect(() => {
    if (!imageSrc) {
      setZoom(1)
      setImageReady(false)
      setNaturalSize({ w: 0, h: 0 })
      return
    }
    setZoom(1)
    setImageReady(false)
    setNaturalSize({ w: 0, h: 0 })
    userClearedMaskRef.current = false
    loadedInitialMaskUrlRef.current = null
    const probe = new Image()
    probe.onload = () => {
      applyImageDimensions(probe.naturalWidth, probe.naturalHeight)
      let canvas = sampleCanvasRef.current
      if (!canvas) {
        canvas = document.createElement('canvas')
        sampleCanvasRef.current = canvas
      }
      canvas.width = probe.naturalWidth
      canvas.height = probe.naturalHeight
      const ctx = canvas.getContext('2d')
      if (ctx) ctx.drawImage(probe, 0, 0)
    }
    probe.onerror = () => setImageReady(false)
    probe.src = imageSrc
  }, [imageSrc, applyImageDimensions])

  useEffect(() => {
    const vp = viewportRef.current
    if (!vp || naturalSize.w <= 0) return
    const ro = new ResizeObserver(() => recomputeFitScale(naturalSize.w, naturalSize.h))
    ro.observe(vp)
    return () => ro.disconnect()
  }, [recomputeFitScale, naturalSize.w, naturalSize.h])

  const exportMask = useCallback(() => {
    const canvas = maskRef.current
    if (!canvas || canvas.width === 0) {
      onMaskChange(null)
      return
    }
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      onMaskChange(null)
      return
    }
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data
    let hasInk = false
    for (let i = 0; i < data.length; i += 4) {
      if (data[i] > 8) {
        hasInk = true
        break
      }
    }
    if (!hasInk) {
      onMaskChange(null)
      return
    }
    canvas.toBlob((blob) => onMaskChange(blob), 'image/png')
  }, [onMaskChange])

  useEffect(() => {
    if (!imageReady || naturalSize.w <= 0 || !initialMaskUrl) return
    if (userClearedMaskRef.current) return
    if (loadedInitialMaskUrlRef.current === initialMaskUrl) return

    const canvas = maskRef.current
    const img = imgRef.current
    if (!canvas || !img?.naturalWidth) return

    let cancelled = false
    loadedInitialMaskUrlRef.current = initialMaskUrl

    void (async () => {
      if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
        canvas.width = img.naturalWidth
        canvas.height = img.naturalHeight
      }
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      try {
        const maskImg = new Image()
        maskImg.crossOrigin = 'anonymous'
        await new Promise<void>((resolve, reject) => {
          maskImg.onload = () => resolve()
          maskImg.onerror = () => reject(new Error('mask load failed'))
          maskImg.src = initialMaskUrl
        })
        if (cancelled || userClearedMaskRef.current) return
        ctx.drawImage(maskImg, 0, 0, canvas.width, canvas.height)
        exportMask()
      } catch {
        if (!cancelled) loadedInitialMaskUrlRef.current = null
      }
    })()

    return () => {
      cancelled = true
    }
  }, [imageReady, naturalSize.w, initialMaskUrl, exportMask])

  useEffect(() => {
    if (!imageReady || naturalSize.w <= 0) return
    const canvas = maskRef.current
    const img = imgRef.current
    if (!canvas || !img?.naturalWidth) return
    if (canvas.width === img.naturalWidth && canvas.height === img.naturalHeight) return
    canvas.width = img.naturalWidth
    canvas.height = img.naturalHeight
    const ctx = canvas.getContext('2d')
    ctx?.clearRect(0, 0, canvas.width, canvas.height)
  }, [imageReady, naturalSize.w, naturalSize.h])

  const syncSampleCanvas = useCallback(() => {
    const img = imgRef.current
    if (!img?.naturalWidth) return
    let canvas = sampleCanvasRef.current
    if (!canvas) {
      canvas = document.createElement('canvas')
      sampleCanvasRef.current = canvas
    }
    canvas.width = img.naturalWidth
    canvas.height = img.naturalHeight
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.drawImage(img, 0, 0)
  }, [])

  const onImgLoad = useCallback(() => {
    const img = imgRef.current
    if (!img?.naturalWidth) return
    applyImageDimensions(img.naturalWidth, img.naturalHeight)
    syncSampleCanvas()
  }, [applyImageDimensions, syncSampleCanvas])

  const sampleColorAtClient = useCallback((clientX: number, clientY: number): string | null => {
    const img = imgRef.current
    const canvas = sampleCanvasRef.current
    if (!img?.naturalWidth || !canvas) return null
    const ctx = canvas.getContext('2d')
    if (!ctx) return null
    const { x, y } = clientToNatural(clientX, clientY, img)
    return normalizeHex(sampleAverageColor(ctx, x, y, 3, canvas.width, canvas.height))
  }, [])

  const handlePickerMove = (e: React.MouseEvent) => {
    if (tool !== 'picker') return
    const hex = sampleColorAtClient(e.clientX, e.clientY)
    if (!hex) return
    setPickerHover({ hex, clientX: e.clientX, clientY: e.clientY })
  }

  const handlePickerLeave = () => {
    setPickerHover(null)
  }

  const handlePickerClick = (e: React.MouseEvent) => {
    if (tool !== 'picker') return
    const hex = sampleColorAtClient(e.clientX, e.clientY)
    if (!hex) return
    setLockedPick({ hex, clientX: e.clientX, clientY: e.clientY })
    setPickerHover(null)
  }

  const addMustIncludeColor = (hex: string) => {
    if (!onMustIncludeColorsChange) return
    const normalized = normalizeHex(hex)
    if (mustIncludeColors.some((c) => normalizeHex(c) === normalized)) {
      setLockedPick(null)
      return
    }
    if (mustIncludeColors.length >= maxMustIncludeColors) return
    onMustIncludeColorsChange([...mustIncludeColors, normalized])
    setLockedPick(null)
  }

  const removeMustIncludeColor = (hex: string) => {
    if (!onMustIncludeColorsChange) return
    const target = normalizeHex(hex)
    onMustIncludeColorsChange(mustIncludeColors.filter((c) => normalizeHex(c) !== target))
  }

  const paintBrush = useCallback(
    (x: number, y: number) => {
      const canvas = maskRef.current
      if (!canvas) return
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      const r = Math.max(4, brushSize)
      const g = ctx.createRadialGradient(x, y, 0, x, y, r)
      g.addColorStop(0, 'rgba(255,255,255,0.95)')
      g.addColorStop(0.65, 'rgba(255,255,255,0.55)')
      g.addColorStop(1, 'rgba(255,255,255,0)')
      ctx.fillStyle = g
      ctx.beginPath()
      ctx.arc(x, y, r, 0, Math.PI * 2)
      ctx.fill()
    },
    [brushSize]
  )

  const fillLasso = useCallback((points: { x: number; y: number }[]) => {
    if (points.length < 3) return
    const canvas = maskRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.fillStyle = 'rgba(255,255,255,0.88)'
    ctx.beginPath()
    ctx.moveTo(points[0].x, points[0].y)
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y)
    ctx.closePath()
    ctx.fill()
  }, [])

  const zoomAtClient = useCallback((clientX: number, clientY: number, factor: number) => {
    const vp = viewportRef.current
    if (!vp || naturalSize.w === 0) return
    const nextZoom = clamp(zoom * factor, MIN_ZOOM, MAX_ZOOM)
    if (nextZoom === zoom) return

    const rect = vp.getBoundingClientRect()
    const cursorX = clientX - rect.left + vp.scrollLeft
    const cursorY = clientY - rect.top + vp.scrollTop
    const ratio = nextZoom / zoom

    setZoom(nextZoom)
    requestAnimationFrame(() => {
      vp.scrollLeft = cursorX * ratio - (clientX - rect.left)
      vp.scrollTop = cursorY * ratio - (clientY - rect.top)
    })
  }, [zoom, naturalSize.w])

  const onViewportWheel = (e: React.WheelEvent) => {
    if (!e.ctrlKey && !e.metaKey) return
    e.preventDefault()
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12
    zoomAtClient(e.clientX, e.clientY, factor)
  }

  const onPointerDown = (e: React.PointerEvent) => {
    if (tool === 'picker') return
    if (e.button !== 0) return
    const img = imgRef.current
    if (!img?.naturalWidth) return
    const { x, y } = clientToNatural(e.clientX, e.clientY, img)
    setDrawing(true)
    if (tool === 'brush') {
      paintBrush(x, y)
    } else {
      const start = [{ x, y }]
      lassoRef.current = start
      setLassoPoints(start)
    }
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
  }

  const onPointerMove = (e: React.PointerEvent) => {
    if (!drawing) return
    const img = imgRef.current
    if (!img?.naturalWidth) return
    const { x, y } = clientToNatural(e.clientX, e.clientY, img)
    if (tool === 'brush') {
      paintBrush(x, y)
    } else {
      lassoRef.current = [...lassoRef.current, { x, y }]
      setLassoPoints(lassoRef.current)
    }
  }

  const onPointerUp = (e: React.PointerEvent) => {
    if (!drawing) return
    setDrawing(false)
    if (tool === 'lasso' && lassoRef.current.length >= 3) {
      fillLasso(lassoRef.current)
    }
    lassoRef.current = []
    setLassoPoints([])
    exportMask()
    try {
      ;(e.target as HTMLElement).releasePointerCapture(e.pointerId)
    } catch {
      /* ignore */
    }
  }

  const clearMask = () => {
    userClearedMaskRef.current = true
    loadedInitialMaskUrlRef.current = null
    const canvas = maskRef.current
    if (canvas) {
      const ctx = canvas.getContext('2d')
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height)
    }
    onMaskChange(null)
  }

  const resetView = () => {
    setZoom(1)
    const vp = viewportRef.current
    if (vp) {
      vp.scrollLeft = 0
      vp.scrollTop = 0
    }
  }

  if (!imageSrc) {
    return (
      <p className="text-sm text-gray-500">Upload an image to mark a priority region on the original.</p>
    )
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2 items-center">
        <button
          type="button"
          onClick={() => {
            setTool('brush')
            setPickerHover(null)
            setLockedPick(null)
          }}
          className={`px-3 py-1.5 text-sm rounded ${tool === 'brush' ? 'bg-teal-600 text-white' : 'bg-gray-700 text-gray-200'}`}
        >
          Brush
        </button>
        <button
          type="button"
          onClick={() => {
            setTool('lasso')
            setPickerHover(null)
            setLockedPick(null)
          }}
          className={`px-3 py-1.5 text-sm rounded ${tool === 'lasso' ? 'bg-teal-600 text-white' : 'bg-gray-700 text-gray-200'}`}
        >
          Lasso
        </button>
        {onMustIncludeColorsChange && (
          <button
            type="button"
            onClick={() => {
              setTool('picker')
              setPickerHover(null)
              setLockedPick(null)
            }}
            className={`px-3 py-1.5 text-sm rounded ${tool === 'picker' ? 'bg-violet-600 text-white' : 'bg-gray-700 text-gray-200'}`}
            title="Pick a colour from the image to force into the palette"
          >
            Pick colour
          </button>
        )}
        <button
          type="button"
          onClick={clearMask}
          className="px-3 py-1.5 text-sm rounded bg-gray-700 text-gray-200 hover:bg-gray-600"
        >
          Clear
        </button>
        <div className="flex items-center gap-1 border border-gray-600 rounded px-1">
          <button
            type="button"
            aria-label="Zoom out"
            className="px-2 py-1 text-sm text-gray-200 hover:bg-gray-700 rounded"
            onClick={() => setZoom((z) => clamp(z / 1.25, MIN_ZOOM, MAX_ZOOM))}
          >
            −
          </button>
          <label className="flex items-center gap-1.5 text-xs text-gray-400 px-1">
            <span className="w-10 text-right tabular-nums">{Math.round(zoom * 100)}%</span>
            <input
              type="range"
              min={MIN_ZOOM}
              max={MAX_ZOOM}
              step={0.05}
              value={zoom}
              onChange={(e) => setZoom(parseFloat(e.target.value))}
              className="w-20 accent-teal-500"
              aria-label="Zoom level"
            />
          </label>
          <button
            type="button"
            aria-label="Zoom in"
            className="px-2 py-1 text-sm text-gray-200 hover:bg-gray-700 rounded"
            onClick={() => setZoom((z) => clamp(z * 1.25, MIN_ZOOM, MAX_ZOOM))}
          >
            +
          </button>
          <button
            type="button"
            onClick={resetView}
            className="px-2 py-1 text-xs text-gray-400 hover:text-gray-200"
          >
            Fit
          </button>
        </div>
        {tool !== 'picker' && (
          <label className="flex items-center gap-2 text-sm text-gray-400">
            Brush
            <input
              type="range"
              min={8}
              max={80}
              value={brushSize}
              onChange={(e) => onBrushSizeChange(parseInt(e.target.value, 10))}
              className="w-20 accent-teal-500"
            />
          </label>
        )}
        <div className="flex items-center gap-1 ml-auto sm:ml-0">
          <span className="text-xs text-gray-500 mr-1">Mask</span>
          {(['show', 'dim', 'hide'] as const).map((mode) => (
            <button
              key={mode}
              type="button"
              onClick={() => setMaskOverlayMode(mode)}
              className={`px-2 py-1 text-xs rounded ${
                maskOverlayMode === mode
                  ? 'bg-teal-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
              title={
                mode === 'show'
                  ? 'Show painted region'
                  : mode === 'dim'
                    ? 'Dim overlay to inspect the photo'
                    : 'Hide overlay — full original visible'
              }
            >
              {mode === 'show' ? 'On' : mode === 'dim' ? 'Dim' : 'Off'}
            </button>
          ))}
        </div>
      </div>

      <div
        ref={viewportRef}
        className="relative w-full overflow-auto rounded-lg border border-gray-600 bg-black"
        style={{ height: viewportHeightPx }}
        onWheel={onViewportWheel}
      >
        {naturalSize.w > 0 && displayW > 0 ? (
          <div className="relative inline-block" style={{ width: displayW, height: displayH }}>
            <img
              ref={imgRef}
              src={imageSrc}
              alt="Draw priority region"
              width={displayW}
              height={displayH}
              className="block select-none pointer-events-none"
              draggable={false}
              onLoad={onImgLoad}
              crossOrigin="anonymous"
            />
            {tool === 'picker' && (
              <div
                className="absolute left-0 top-0 touch-none cursor-crosshair"
                style={{ width: displayW, height: displayH }}
                onMouseMove={handlePickerMove}
                onMouseLeave={handlePickerLeave}
                onClick={handlePickerClick}
                role="presentation"
              />
            )}
            <canvas
              ref={maskRef}
              className="absolute left-0 top-0 touch-none cursor-crosshair"
              style={{
                width: displayW,
                height: displayH,
                mixBlendMode: 'screen',
                opacity: MASK_OVERLAY_OPACITY[maskOverlayMode],
                pointerEvents: tool === 'picker' ? 'none' : 'auto',
              }}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerLeave={onPointerUp}
            />
            {tool === 'lasso' && lassoPoints.length > 1 && naturalSize.w > 0 && (
              <svg
                className="absolute left-0 top-0 pointer-events-none"
                width={displayW}
                height={displayH}
                viewBox={`0 0 ${naturalSize.w} ${naturalSize.h}`}
                preserveAspectRatio="none"
              >
                <polyline
                  points={lassoPoints.map((p) => `${p.x},${p.y}`).join(' ')}
                  fill="none"
                  stroke="rgb(45, 212, 191)"
                  strokeWidth={Math.max(2, brushSize * 0.15)}
                />
              </svg>
            )}
          </div>
        ) : (
          <p className="text-sm text-gray-500 p-4">Loading image…</p>
        )}
        {viewportRef.current && (() => {
          const vp = viewportRef.current!
          const rect = vp.getBoundingClientRect()
          const popupPosition = (clientX: number, clientY: number, boxW: number, boxH: number) => ({
            left: Math.min(
              Math.max(8, clientX - rect.left + vp.scrollLeft + 14),
              Math.max(8, rect.width - boxW - 8)
            ),
            top: Math.min(
              Math.max(8, clientY - rect.top + vp.scrollTop + 14),
              Math.max(8, viewportHeightPx - boxH - 8)
            ),
          })

          return (
            <>
              {tool === 'picker' && pickerHover && !lockedPick && (
                <div
                  className="absolute z-20 pointer-events-none rounded-lg border border-gray-600/80 bg-gray-900/95 shadow-lg px-2.5 py-2 flex items-center gap-2.5"
                  style={popupPosition(pickerHover.clientX, pickerHover.clientY, 140, 48)}
                >
                  <div
                    className="w-9 h-9 rounded border border-gray-500 shrink-0"
                    style={{ backgroundColor: pickerHover.hex }}
                  />
                  <div>
                    <p className="font-mono text-xs text-gray-100">{pickerHover.hex}</p>
                    <p className="text-[10px] text-gray-500">Click to pick</p>
                  </div>
                </div>
              )}
              {lockedPick && (() => {
                const pos = popupPosition(lockedPick.clientX, lockedPick.clientY, 200, 120)
                const alreadyIncluded = mustIncludeColors.some(
                  (c) => normalizeHex(c) === lockedPick.hex
                )
                const atLimit = mustIncludeColors.length >= maxMustIncludeColors
                return (
                  <div
                    className="absolute z-30 rounded-lg border border-violet-500/60 bg-gray-900 shadow-xl p-3 min-w-[11rem]"
                    style={pos}
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <div
                        className="w-12 h-12 rounded border-2 border-violet-400 shrink-0"
                        style={{ backgroundColor: lockedPick.hex }}
                      />
                      <div>
                        <p className="text-xs text-violet-300">Locked</p>
                        <p className="font-mono text-sm text-gray-100">{lockedPick.hex}</p>
                      </div>
                    </div>
                    {alreadyIncluded ? (
                      <p className="text-xs text-amber-400 mb-2">Already in must-include list.</p>
                    ) : atLimit ? (
                      <p className="text-xs text-amber-400 mb-2">
                        Maximum {maxMustIncludeColors} must-include colours for this palette size.
                      </p>
                    ) : null}
                    <div className="flex gap-2">
                      <button
                        type="button"
                        disabled={alreadyIncluded || atLimit}
                        onClick={() => addMustIncludeColor(lockedPick.hex)}
                        className="flex-1 px-2 py-1.5 text-sm rounded bg-violet-600 hover:bg-violet-500 text-white disabled:opacity-40 disabled:cursor-not-allowed"
                      >
                        Must include
                      </button>
                      <button
                        type="button"
                        onClick={() => setLockedPick(null)}
                        className="px-2 py-1.5 text-sm rounded bg-gray-700 hover:bg-gray-600 text-gray-200"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )
              })()}
            </>
          )
        })()}
      </div>

      {onMustIncludeColorsChange && mustIncludeColors.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-gray-400 uppercase tracking-wide">
            Must-include colours ({mustIncludeColors.length})
          </p>
          <div className="flex flex-wrap gap-2">
            {mustIncludeColors.map((hex) => (
              <div
                key={hex}
                className="flex items-center gap-2 pl-1 pr-2 py-1 rounded-full border border-violet-700/60 bg-violet-950/30"
              >
                <div
                  className="w-7 h-7 rounded-full border border-gray-600 shrink-0"
                  style={{ backgroundColor: hex }}
                  title={hex}
                />
                <span className="font-mono text-xs text-gray-200">{hex}</span>
                <button
                  type="button"
                  onClick={() => removeMustIncludeColor(hex)}
                  className="ml-0.5 w-5 h-5 rounded-full text-gray-400 hover:text-white hover:bg-gray-700 text-sm leading-none"
                  aria-label={`Remove ${hex}`}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      <p className="text-xs text-gray-500">
        {tool === 'picker' ? (
          <>
            Move over the image to preview a colour; click to lock, then choose{' '}
            <strong className="text-gray-400">Must include</strong>.
            Scroll to pan; Ctrl+scroll zooms.
          </>
        ) : (
          <>
            Scroll to pan when zoomed in. Ctrl+scroll (or Cmd+scroll on Mac) zooms toward the cursor. Use{' '}
            <strong className="text-gray-400">Mask Off</strong> to see the full photo; drawing still updates the hidden
            region.
          </>
        )}
      </p>

      {typeof detailInRegion === 'number' && onDetailInRegionChange && (
        <div>
          <label className="block text-sm mb-2">
            Detail in region: {(detailInRegion * 100).toFixed(0)}%
          </label>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={detailInRegion}
            onChange={(e) => onDetailInRegionChange(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-teal-500"
          />
          <p className="text-xs text-gray-500 mt-1">Try 80–100% for faces. Pair with Favor skin tones below.</p>
        </div>
      )}
    </div>
  )
}
