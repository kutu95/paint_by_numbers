'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { useParams } from 'next/navigation'
import { API_BASE_URL } from '@/lib/config'
import type { Layer, SessionData } from '../types'
import { loadViewerHudState, saveViewerHudState, getViewerHudKey } from './projectionViewerState'
import {
  getLassoMode,
  setLassoMode,
  getLassoPath,
  setLassoPath,
  getLassoModeKey,
  getLassoPathKey,
  type LassoPoint,
} from './projectionLassoState'

type OutlineMode = 'off' | 'thin' | 'thick' | 'glow'

const PROJECTION_LAYER_KEY = (id: string) => `projection_current_layer_${id}`

function pointInPolygon(px: number, py: number, path: { x: number; y: number }[]): boolean {
  if (path.length < 3) return false
  let inside = false
  const n = path.length
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = path[i].x, yi = path[i].y
    const xj = path[j].x, yj = path[j].y
    if (((yi > py) !== (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) inside = !inside
  }
  return inside
}

export default function ProjectionViewerWindow() {
  const params = useParams()
  const sessionId = params.sessionId as string
  const [sessionData, setSessionData] = useState<SessionData | null>(null)
  const [currentLayer, setCurrentLayerState] = useState(0)
  const setCurrentLayer = useCallback((n: number) => {
    setCurrentLayerState(n)
    if (sessionId) localStorage.setItem(PROJECTION_LAYER_KEY(sessionId), String(n))
  }, [sessionId])
  const [crosshairs, setCrosshairs] = useState(true)
  const [grid, setGrid] = useState(false)
  const [inverted, setInverted] = useState(false)
  const [showColor, setShowColor] = useState(false)
  const [outlineMode, setOutlineMode] = useState<OutlineMode>('off')
  const [maskOpacity, setMaskOpacity] = useState(85)
  const [registrationMode, setRegistrationMode] = useState(false)
  const [blackScreen, setBlackScreen] = useState(false)
  const [whiteScreen, setWhiteScreen] = useState(false)
  const [mouseActive, setMouseActive] = useState(true)
  const [doneLayers, setDoneLayers] = useState<Set<number>>(new Set())
  const [showDoneLayers, setShowDoneLayers] = useState(false)
  const [projectionScale, setProjectionScale] = useState(1.0)
  const [showFinalPreview, setShowFinalPreview] = useState(false)
  const [showOriginalImage, setShowOriginalImage] = useState(false)
  const [usePureMask, setUsePureMask] = useState(false)
  const [maskLoadError, setMaskLoadError] = useState<string | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const imageContainerRef = useRef<HTMLDivElement>(null)
  const mouseTimerRef = useRef<NodeJS.Timeout>()
  const touchStartRef = useRef<{ x: number; y: number; time: number } | null>(null)
  const [coloredMaskUrl, setColoredMaskUrl] = useState<string | null>(null)
  const [lassoMode, setLassoModeState] = useState<'' | 'drawing' | 'active'>(() => (typeof window !== 'undefined' ? getLassoMode(sessionId) : ''))
  const [lassoPath, setLassoPathState] = useState<LassoPoint[] | null>(() => (typeof window !== 'undefined' ? getLassoPath(sessionId) : null))
  const [lassoDrawingPoints, setLassoDrawingPoints] = useState<LassoPoint[]>([])
  const [layersInLasso, setLayersInLasso] = useState<number[]>([])
  const lassoPathComputedRef = useRef(false)

  useEffect(() => {
    const stored = localStorage.getItem(`session_${sessionId}`)
    if (stored) {
      try {
        const data = JSON.parse(stored) as SessionData
        setSessionData(data)
        const done = localStorage.getItem(`done_${sessionId}`)
        if (done) setDoneLayers(new Set(JSON.parse(done)))
        const layerStored = localStorage.getItem(PROJECTION_LAYER_KEY(sessionId))
        if (layerStored !== null) {
          const n = parseInt(layerStored, 10)
          if (!Number.isNaN(n) && n >= 0) setCurrentLayerState(n)
        }
        const hud = loadViewerHudState(sessionId)
        setCrosshairs(hud.crosshairs)
        setGrid(hud.grid)
        setInverted(hud.inverted)
        setShowColor(hud.showColor)
        setOutlineMode(hud.outlineMode)
        setMaskOpacity(hud.maskOpacity)
        setRegistrationMode(hud.registrationMode)
        setBlackScreen(hud.blackScreen)
        setWhiteScreen(hud.whiteScreen)
        setShowDoneLayers(hud.showDoneLayers)
        setShowFinalPreview(hud.showFinalPreview)
        setShowOriginalImage(hud.showOriginalImage)
        setUsePureMask(hud.usePureMask)
      } catch (e) {
        console.error('Failed to load session data')
      }
    }
  }, [sessionId])

  useEffect(() => {
    const key = getViewerHudKey(sessionId)
    const onStorage = (e: StorageEvent) => {
      if (e.key === key && e.newValue !== null) {
        try {
          const hud = JSON.parse(e.newValue) as Partial<ReturnType<typeof loadViewerHudState>>
          if (hud.crosshairs !== undefined) setCrosshairs(hud.crosshairs)
          if (hud.grid !== undefined) setGrid(hud.grid)
          if (hud.inverted !== undefined) setInverted(hud.inverted)
          if (hud.showColor !== undefined) setShowColor(hud.showColor)
          if (hud.outlineMode !== undefined) setOutlineMode(hud.outlineMode)
          if (hud.maskOpacity !== undefined) setMaskOpacity(hud.maskOpacity)
          if (hud.registrationMode !== undefined) setRegistrationMode(hud.registrationMode)
          if (hud.blackScreen !== undefined) setBlackScreen(hud.blackScreen)
          if (hud.whiteScreen !== undefined) setWhiteScreen(hud.whiteScreen)
          if (hud.showDoneLayers !== undefined) setShowDoneLayers(hud.showDoneLayers)
          if (hud.showFinalPreview !== undefined) setShowFinalPreview(hud.showFinalPreview)
          if (hud.showOriginalImage !== undefined) setShowOriginalImage(hud.showOriginalImage)
          if (hud.usePureMask !== undefined) setUsePureMask(hud.usePureMask)
        } catch (_) {}
      }
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [sessionId])

  useEffect(() => {
    if (!sessionId) return
    saveViewerHudState(sessionId, {
      crosshairs,
      grid,
      inverted,
      showColor,
      outlineMode,
      maskOpacity,
      registrationMode,
      blackScreen,
      whiteScreen,
      showDoneLayers,
      showFinalPreview,
      showOriginalImage,
      usePureMask,
    })
  }, [sessionId, crosshairs, grid, inverted, showColor, outlineMode, maskOpacity, registrationMode, blackScreen, whiteScreen, showDoneLayers, showFinalPreview, showOriginalImage, usePureMask])

  useEffect(() => {
    const key = PROJECTION_LAYER_KEY(sessionId)
    const onStorage = (e: StorageEvent) => {
      if (e.key === key && e.newValue !== null) {
        const n = parseInt(e.newValue, 10)
        if (!Number.isNaN(n) && n >= 0) setCurrentLayerState(n)
      }
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [sessionId])

  useEffect(() => {
    setLassoModeState(getLassoMode(sessionId))
    setLassoPathState(getLassoPath(sessionId))
  }, [sessionId])

  useEffect(() => {
    const modeKey = getLassoModeKey(sessionId)
    const pathKey = getLassoPathKey(sessionId)
    const onStorage = (e: StorageEvent) => {
      if (e.key === modeKey) {
        const v = e.newValue
        setLassoModeState(v === 'drawing' || v === 'active' ? v : '')
        if (v !== 'drawing' && v !== 'active') {
          setLassoPathState(null)
          setLassoDrawingPoints([])
        }
      }
      if (e.key === pathKey && e.newValue) {
        try {
          setLassoPathState(JSON.parse(e.newValue) as LassoPoint[])
        } catch (_) {}
      }
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [sessionId])

  useEffect(() => {
    const saved = localStorage.getItem(`projection_scale_${sessionId}`)
    if (saved != null) {
      const n = parseFloat(saved)
      if (!Number.isNaN(n) && n >= 0.25 && n <= 2) setProjectionScale(n)
    }
  }, [sessionId])

  useEffect(() => {
    if (projectionScale >= 0.25 && projectionScale <= 2) {
      localStorage.setItem(`projection_scale_${sessionId}`, String(projectionScale))
    }
  }, [sessionId, projectionScale])

  useEffect(() => {
    const scaleKey = `projection_scale_${sessionId}`
    const onStorage = (e: StorageEvent) => {
      if (e.key === scaleKey && e.newValue != null) {
        const n = parseFloat(e.newValue)
        if (!Number.isNaN(n) && n >= 0.25 && n <= 2) setProjectionScale(n)
      }
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [sessionId])

  const saveDoneLayers = useCallback((layers: Set<number>) => {
    localStorage.setItem(`done_${sessionId}`, JSON.stringify(Array.from(layers)))
  }, [sessionId])

  const navigateLayer = useCallback((direction: number) => {
    if (!sessionData) return
    const maxLayer = sessionData.layers.length - 1
    if (lassoMode === 'active' && layersInLasso.length > 0) {
      const idx = layersInLasso.indexOf(currentLayer)
      let nextIdx = idx + direction
      if (nextIdx < 0) nextIdx = layersInLasso.length - 1
      if (nextIdx >= layersInLasso.length) nextIdx = 0
      setCurrentLayer(layersInLasso[nextIdx]!)
      return
    }
    let next = currentLayer + direction
    if (!showDoneLayers) {
      while (next >= 0 && next <= maxLayer && doneLayers.has(next) && !sessionData.layers[next]?.is_finished) {
        next += direction
      }
    }
    if (next >= 0 && next <= maxLayer) setCurrentLayer(next)
  }, [sessionData, currentLayer, doneLayers, showDoneLayers, setCurrentLayer, lassoMode, layersInLasso])

  useEffect(() => {
    if (!sessionData || lassoMode !== 'active' || !lassoPath || lassoPath.length < 3) {
      setLayersInLasso([])
      lassoPathComputedRef.current = false
      return
    }
    let cancelled = false
    const baseUrl = API_BASE_URL
    const checkLayer = async (layerIndex: number): Promise<boolean> => {
      const layer = sessionData.layers[layerIndex]
      if (!layer?.mask_url) return false
      const url = `${baseUrl}${usePureMask && layer.mask_pure_url ? layer.mask_pure_url : layer.mask_url}`
      return new Promise((resolve) => {
        const img = new Image()
        img.crossOrigin = 'anonymous'
        img.onload = () => {
          if (cancelled) { resolve(false); return }
          const w = img.width
          const h = img.height
          const canvas = document.createElement('canvas')
          canvas.width = w
          canvas.height = h
          const ctx = canvas.getContext('2d')
          if (!ctx) { resolve(false); return }
          ctx.drawImage(img, 0, 0)
          const data = ctx.getImageData(0, 0, w, h).data
          const pathPx = lassoPath!.map((p) => ({ x: p.x * w, y: p.y * h }))
          const minX = Math.max(0, Math.floor(Math.min(...pathPx.map((p) => p.x))))
          const maxX = Math.min(w - 1, Math.ceil(Math.max(...pathPx.map((p) => p.x))))
          const minY = Math.max(0, Math.floor(Math.min(...pathPx.map((p) => p.y))))
          const maxY = Math.min(h - 1, Math.ceil(Math.max(...pathPx.map((p) => p.y))))
          for (let py = minY; py <= maxY && !cancelled; py++) {
            for (let px = minX; px <= maxX; px++) {
              const nx = px / w
              const ny = py / h
              if (pointInPolygon(nx, ny, lassoPath!)) {
                const i = (py * w + px) * 4
                if (data[i]! > 0 || data[i + 1]! > 0 || data[i + 2]! > 0) {
                  resolve(true)
                  return
                }
              }
            }
          }
          resolve(false)
        }
        img.onerror = () => resolve(false)
        img.src = url
      })
    }
    ;(async () => {
      const visible: number[] = []
      for (let i = 0; i < sessionData.layers.length && !cancelled; i++) {
        if (sessionData.layers[i]?.is_finished) continue
        const hasContent = await checkLayer(i)
        if (hasContent) visible.push(i)
      }
      if (!cancelled) {
        setLayersInLasso(visible)
        lassoPathComputedRef.current = true
      }
    })()
    return () => { cancelled = true }
  }, [sessionData, lassoMode, lassoPath, usePureMask])

  useEffect(() => {
    if (lassoMode === 'active' && layersInLasso.length > 0 && !layersInLasso.includes(currentLayer)) {
      setCurrentLayer(layersInLasso[0]!)
    }
  }, [lassoMode, layersInLasso, currentLayer, setCurrentLayer])

  const toggleDone = useCallback(() => {
    if (sessionData?.layers[currentLayer]?.is_finished) return
    const newDone = new Set(doneLayers)
    if (newDone.has(currentLayer)) newDone.delete(currentLayer)
    else newDone.add(currentLayer)
    setDoneLayers(newDone)
    saveDoneLayers(newDone)
  }, [currentLayer, doneLayers, saveDoneLayers, sessionData])

  useEffect(() => {
    const handleMouseMove = () => {
      setMouseActive(true)
      if (mouseTimerRef.current) clearTimeout(mouseTimerRef.current)
      mouseTimerRef.current = setTimeout(() => setMouseActive(false), 2000)
    }
    window.addEventListener('mousemove', handleMouseMove)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      if (mouseTimerRef.current) clearTimeout(mouseTimerRef.current)
    }
  }, [])

  useEffect(() => {
    const handleTouchStart = (e: TouchEvent) => {
      const t = e.touches[0]
      touchStartRef.current = { x: t.clientX, y: t.clientY, time: Date.now() }
    }
    const handleTouchEnd = (e: TouchEvent) => {
      if (!touchStartRef.current) return
      const t = e.changedTouches[0]
      const dx = t.clientX - touchStartRef.current.x
      const dy = t.clientY - touchStartRef.current.y
      const dt = Date.now() - touchStartRef.current.time
      if (Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > 50 && dt < 500) {
        navigateLayer(dx < 0 ? 1 : -1)
      }
      touchStartRef.current = null
    }
    const el = containerRef.current
    if (el) {
      el.addEventListener('touchstart', handleTouchStart, { passive: true })
      el.addEventListener('touchend', handleTouchEnd, { passive: true })
      return () => {
        el.removeEventListener('touchstart', handleTouchStart)
        el.removeEventListener('touchend', handleTouchEnd)
      }
    }
  }, [navigateLayer])

  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT') return
      switch (e.key.toLowerCase()) {
        case 'c': setCrosshairs((p) => !p); break
        case 'g':
          e.preventDefault()
          setShowOriginalImage((p) => { if (!p) setShowFinalPreview(false); return !p })
          break
        case 'i': setInverted((p) => { if (!p) setShowColor(false); return !p }); break
        case 'k': setShowColor((p) => { if (!p) setInverted(false); return !p }); break
        case 'o':
          setOutlineMode((p) => (['off', 'thin', 'thick', 'glow'] as OutlineMode[])[((['off', 'thin', 'thick', 'glow'].indexOf(p)) + 1) % 4])
          break
        case '[': setMaskOpacity((p) => Math.max(40, p - 5)); break
        case ']': setMaskOpacity((p) => Math.min(100, p + 5)); break
        case 'r': setRegistrationMode((p) => !p); break
        case 'b': setBlackScreen((p) => !p); setWhiteScreen(false); break
        case 'w': setWhiteScreen((p) => !p); setBlackScreen(false); break
        case 'h': break // HUD controls are on the Projection tab
        case 'arrowleft': navigateLayer(-1); break
        case 'arrowright': navigateLayer(1); break
        case ' ': e.preventDefault(); navigateLayer(1); break
        case 'd': toggleDone(); break
        case 's': setShowDoneLayers((p) => !p); break
        case '-': setProjectionScale((p) => Math.max(0.25, Math.round((p - 0.05) * 100) / 100)); break
        case '=':
        case '+': setProjectionScale((p) => Math.min(2, Math.round((p + 0.05) * 100) / 100)); break
        case 'f':
          e.preventDefault()
          setShowFinalPreview((p) => { if (!p) setShowOriginalImage(false); return !p })
          break
        case 'l': setUsePureMask((p) => !p); break
        case 'x': setGrid((p) => !p); break
        case 'e':
          if (lassoMode === 'active' || lassoMode === 'drawing') {
            setLassoMode(sessionId, '')
            setLassoModeState('')
            setLassoPathState(null)
            setLassoDrawingPoints([])
          }
          break
        case 'enter':
          if (lassoMode === 'drawing' && lassoDrawingPoints.length >= 3) {
            setLassoPath(sessionId, lassoDrawingPoints)
            setLassoPathState(lassoDrawingPoints)
            setLassoMode(sessionId, 'active')
            setLassoModeState('active')
            setLassoDrawingPoints([])
          }
          break
        case 'escape':
        case 'Escape':
          if (lassoMode === 'drawing') {
            setLassoMode(sessionId, '')
            setLassoModeState('')
            setLassoDrawingPoints([])
          } else {
            try { window.close() } catch (_) {}
          }
          break
      }
    }
    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [navigateLayer, toggleDone, showDoneLayers, lassoMode, sessionId, lassoDrawingPoints])

  useEffect(() => {
    if (!showColor || !sessionData || currentLayer < 0 || currentLayer >= sessionData.layers.length) {
      setColoredMaskUrl(null)
      return
    }
    const layerData = sessionData.layers[currentLayer]
    if (!layerData || layerData.is_finished) { setColoredMaskUrl(null); return }
    let colorHex: string | null = null
    if (layerData.is_gradient && layerData.hex) colorHex = layerData.hex
    else {
      const pc = sessionData.palette.find((p) => p.index === layerData.palette_index)
      if (pc?.hex) colorHex = pc.hex
    }
    if (!colorHex || !layerData.mask_url) { setColoredMaskUrl(null); return }
    const pureMaskUrl = layerData.mask_pure_url ?? `/api/sessions/${sessionData.session_id}/layer_${layerData.layer_index}_pure_mask.png`
    const maskUrlForColor = usePureMask ? pureMaskUrl : layerData.mask_url
    const paintMaskWithColor = (image: HTMLImageElement) => {
      try {
        const canvas = document.createElement('canvas')
        canvas.width = image.width
        canvas.height = image.height
        const ctx = canvas.getContext('2d')
        if (!ctx) { setColoredMaskUrl(null); return }
        ctx.imageSmoothingEnabled = false
        ctx.drawImage(image, 0, 0)
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
        const data = imageData.data
        const r = parseInt(colorHex!.slice(1, 3), 16)
        const g = parseInt(colorHex!.slice(3, 5), 16)
        const b = parseInt(colorHex!.slice(5, 7), 16)
        for (let i = 0; i < data.length; i += 4) {
          // Treat any non-black mask pixel as paintable to avoid threshold holes.
          if (data[i] > 0 || data[i + 1] > 0 || data[i + 2] > 0) {
            data[i] = r
            data[i + 1] = g
            data[i + 2] = b
          }
        }
        ctx.putImageData(imageData, 0, 0)
        setColoredMaskUrl(canvas.toDataURL())
        setMaskLoadError(null)
      } catch (_) {
        setColoredMaskUrl(null)
      }
    }
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => paintMaskWithColor(img)
    img.onerror = () => {
      setColoredMaskUrl(null)
      setMaskLoadError(usePureMask ? 'Pure mask failed to load' : 'Mask failed to load')
    }
    img.src = `${API_BASE_URL}${maskUrlForColor}`
  }, [showColor, currentLayer, sessionData, API_BASE_URL, usePureMask])

  useEffect(() => { setMaskLoadError(null) }, [currentLayer, usePureMask])

  const handleLassoMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (lassoMode !== 'drawing' || !imageContainerRef.current) return
      const rect = imageContainerRef.current.getBoundingClientRect()
      const x = (e.clientX - rect.left) / rect.width
      const y = (e.clientY - rect.top) / rect.height
      const nx = Math.max(0, Math.min(1, x))
      const ny = Math.max(0, Math.min(1, y))
      setLassoDrawingPoints((prev) => [...prev, { x: nx, y: ny }])
    },
    [lassoMode]
  )

  const handleLassoDoubleClick = useCallback(() => {
    if (lassoMode !== 'drawing' || lassoDrawingPoints.length < 3) return
    setLassoPath(sessionId, lassoDrawingPoints)
    setLassoPathState(lassoDrawingPoints)
    setLassoMode(sessionId, 'active')
    setLassoModeState('active')
    setLassoDrawingPoints([])
  }, [lassoMode, lassoDrawingPoints, sessionId])

  if (!sessionData) {
    return (
      <div className="fixed inset-0 bg-black flex items-center justify-center text-white">
        Loading session...
      </div>
    )
  }

  const currentLayerData = sessionData.layers[currentLayer]
  if (!currentLayerData) {
    return (
      <div className="fixed inset-0 bg-black flex items-center justify-center text-white">
        Invalid layer
      </div>
    )
  }

  let layerColor: { hex: string } | undefined
  if (currentLayerData.is_gradient && currentLayerData.hex) layerColor = { hex: currentLayerData.hex }
  else layerColor = sessionData.palette.find((p) => p.index === currentLayerData.palette_index)

  const baseUrl = API_BASE_URL
  const originalImageUrl = sessionData.original_url ? `${baseUrl}${sessionData.original_url}` : null
  const outlineUrl =
    currentLayerData.is_finished || outlineMode === 'off'
      ? null
      : `${baseUrl}${currentLayerData[`outline_${outlineMode}_url` as keyof Layer]}`
  const finishedLayer = sessionData.layers.find((l) => l.is_finished)
  const finalPreviewUrl = finishedLayer
    ? `${baseUrl}${finishedLayer.finished_url || finishedLayer.mask_url}`
    : sessionData.quantized_preview_url
      ? `${baseUrl}${sessionData.quantized_preview_url}`
      : null

  return (
    <div
      ref={containerRef}
      id="projection-viewer"
      className={`fixed inset-0 bg-black ${mouseActive ? 'show-cursor' : ''}`}
      style={{ cursor: mouseActive ? 'default' : 'none' }}
    >
      {blackScreen && <div className="fixed inset-0 bg-black z-50" />}
      {whiteScreen && <div className="fixed inset-0 bg-white z-50" />}

      {!blackScreen && !whiteScreen && (
        <>
          <button
            type="button"
            onClick={() => { try { window.close() } catch (_) {} }}
            className="fixed top-4 left-4 z-[60] px-4 py-2 bg-black/70 hover:bg-black/90 text-white rounded"
            style={{ opacity: mouseActive ? 1 : 0.3 }}
          >
            Close window
          </button>

          <div className="relative w-full h-full flex items-center justify-center overflow-hidden">
            <div
              ref={imageContainerRef}
              className="w-full h-full flex items-center justify-center"
              style={{
                transform: `scale(${projectionScale})`,
                transformOrigin: 'center center',
                clipPath: lassoMode === 'active' && lassoPath && lassoPath.length >= 3
                  ? `polygon(${lassoPath.map((p) => `${p.x * 100}% ${p.y * 100}%`).join(', ')})`
                  : undefined,
              }}
            >
              {showOriginalImage && originalImageUrl ? (
                <img
                  src={originalImageUrl}
                  alt="Original"
                  className="absolute"
                  style={{
                    opacity: registrationMode ? 0 : 1,
                    filter: inverted ? 'invert(1)' : 'none',
                    maxWidth: '100%',
                    maxHeight: '100%',
                    objectFit: 'contain',
                  }}
                />
              ) : currentLayerData.is_finished || showFinalPreview ? (
                <img
                  src={showFinalPreview && finalPreviewUrl ? finalPreviewUrl : `${baseUrl}${currentLayerData.finished_url || currentLayerData.mask_url}`}
                  alt="Final"
                  className="absolute"
                  style={{
                    opacity: registrationMode ? 0 : 1,
                    filter: inverted ? 'invert(1)' : 'none',
                    maxWidth: '100%',
                    maxHeight: '100%',
                    objectFit: 'contain',
                  }}
                />
              ) : (
                <>
                  {showColor && layerColor && coloredMaskUrl ? (
                    <img
                      src={coloredMaskUrl}
                      alt={`Layer ${currentLayer + 1}`}
                      className="absolute"
                      style={{ opacity: registrationMode ? 0 : maskOpacity / 100, maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
                    />
                  ) : (
                    <img
                      src={`${baseUrl}${usePureMask ? (currentLayerData.mask_pure_url ?? `/api/sessions/${sessionData.session_id}/layer_${currentLayerData.layer_index}_pure_mask.png`) : currentLayerData.mask_url}`}
                      alt={`Layer ${currentLayer + 1}`}
                      className="absolute"
                      crossOrigin="anonymous"
                      onLoad={() => setMaskLoadError(null)}
                      onError={() => setMaskLoadError(usePureMask ? 'Pure mask failed to load' : 'Mask failed to load')}
                      style={{
                        opacity: registrationMode ? 0 : maskOpacity / 100,
                        filter: inverted ? 'invert(1)' : 'none',
                        maxWidth: '100%',
                        maxHeight: '100%',
                        objectFit: 'contain',
                        display: maskLoadError ? 'none' : undefined,
                      }}
                    />
                  )}
                  {maskLoadError && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/80 text-red-400 p-4">
                      {maskLoadError}
                    </div>
                  )}
                  {outlineUrl && (
                    <img
                      src={outlineUrl}
                      alt="Outline"
                      className="absolute pointer-events-none"
                      style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
                    />
                  )}
                </>
              )}
            </div>

            {lassoMode === 'drawing' && (
              <div
                className="absolute inset-0 z-40 flex items-center justify-center"
                style={{ pointerEvents: 'auto' }}
                onMouseDown={handleLassoMouseDown}
                onDoubleClick={handleLassoDoubleClick}
              >
                <div className="absolute inset-0 bg-black/20" />
                <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none">
                  {lassoDrawingPoints.length > 0 && (
                    <polyline
                      fill="none"
                      stroke="rgba(255,255,255,0.9)"
                      strokeWidth="0.5"
                      points={lassoDrawingPoints.map((p) => `${p.x * 100},${p.y * 100}`).join(' ')}
                    />
                  )}
                </svg>
                {lassoDrawingPoints.map((p, i) => (
                  <div
                    key={i}
                    className="absolute w-1.5 h-1.5 rounded-full bg-white pointer-events-none border border-gray-800"
                    style={{
                      left: `${p.x * 100}%`,
                      top: `${p.y * 100}%`,
                      transform: 'translate(-50%, -50%)',
                    }}
                  />
                ))}
                <p className="absolute bottom-4 left-1/2 -translate-x-1/2 text-white bg-black/70 px-4 py-2 rounded text-sm">
                  Click to add points. Double-click or press Enter to close lasso. Esc to cancel. E to end lasso.
                </p>
              </div>
            )}
          </div>

          {crosshairs && (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <div className="w-px h-full bg-white opacity-30" />
              <div className="absolute w-full h-px bg-white opacity-30" />
            </div>
          )}

          {grid && (
            <svg className="pointer-events-none absolute inset-0 w-full h-full text-white opacity-20" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <style>{`.grid-line { stroke: currentColor; stroke-width: 1; }`}</style>
              </defs>
              {Array.from({ length: 20 }).map((_, i) => (
                <g key={i}>
                  <line x1={`${(i + 1) * 5}%`} y1="0%" x2={`${(i + 1) * 5}%`} y2="100%" className="grid-line" />
                  <line x1="0%" y1={`${(i + 1) * 5}%`} x2="100%" y2={`${(i + 1) * 5}%`} className="grid-line" />
                </g>
              ))}
            </svg>
          )}
        </>
      )}
    </div>
  )
}
