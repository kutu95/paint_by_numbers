'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { useParams } from 'next/navigation'
import { projectAssetUrl } from '@/lib/projectAssets'
import { fetchProjectSession, fetchProjectState, saveProjectState } from '@/lib/projectSession'
import type { MaskDisplayMode, ProjectionViewerHudState } from './projectionViewerState'
import {
  buildColorMaskDataUrl,
  buildDetailMaskDataUrl,
  cycleMaskDisplayMode,
  maskDisplayModeLabel,
  resolveMaskDisplayMode,
} from './maskDisplay'
import { PROJECTION_SHORTCUTS_LINES } from './projectionKeyboardHelp'
import { writeProjectionPopupBounds } from '@/lib/projectionWindowBounds'
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
  }, [])
  const [crosshairs, setCrosshairs] = useState(true)
  const [grid, setGrid] = useState(false)
  const [inverted, setInverted] = useState(false)
  const [maskDisplayMode, setMaskDisplayMode] = useState<MaskDisplayMode>('white')
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
  const [showHudOverlay, setShowHudOverlay] = useState(false)
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
    let debounce: ReturnType<typeof setTimeout> | undefined
    const scheduleSave = () => {
      if (debounce !== undefined) clearTimeout(debounce)
      debounce = setTimeout(() => writeProjectionPopupBounds(), 250)
    }
    const saveNow = () => writeProjectionPopupBounds()
    scheduleSave()
    window.addEventListener('move', scheduleSave)
    window.addEventListener('resize', scheduleSave)
    window.addEventListener('beforeunload', saveNow)
    window.addEventListener('pagehide', saveNow)
    const onVis = () => {
      if (document.visibilityState === 'hidden') saveNow()
    }
    document.addEventListener('visibilitychange', onVis)
    return () => {
      if (debounce !== undefined) clearTimeout(debounce)
      window.removeEventListener('move', scheduleSave)
      window.removeEventListener('resize', scheduleSave)
      window.removeEventListener('beforeunload', saveNow)
      window.removeEventListener('pagehide', saveNow)
      document.removeEventListener('visibilitychange', onVis)
      saveNow()
    }
  }, [])

  useEffect(() => {
    if (!sessionId) return
    let cancelled = false
    void (async () => {
      const [session, ui] = await Promise.all([
        fetchProjectSession(sessionId),
        fetchProjectState(sessionId),
      ])
      if (cancelled) return
      if (session) setSessionData(session)
      if (Array.isArray(ui.doneLayers)) setDoneLayers(new Set(ui.doneLayers))
      if (typeof ui.currentLayer === 'number' && ui.currentLayer >= 0) {
        setCurrentLayerState(ui.currentLayer)
      }
      const hud = (ui.projectionHud ?? {}) as Partial<ProjectionViewerHudState>
      setCrosshairs(hud.crosshairs ?? true)
      setGrid(hud.grid ?? false)
      setInverted(hud.inverted ?? false)
      setMaskDisplayMode(resolveMaskDisplayMode(hud))
      setOutlineMode(hud.outlineMode ?? 'off')
      setMaskOpacity(hud.maskOpacity ?? 85)
      setRegistrationMode(hud.registrationMode ?? false)
      setBlackScreen(hud.blackScreen ?? false)
      setWhiteScreen(hud.whiteScreen ?? false)
      setShowDoneLayers(hud.showDoneLayers ?? false)
      setShowFinalPreview(hud.showFinalPreview ?? false)
      setShowOriginalImage(hud.showOriginalImage ?? false)
      setUsePureMask(hud.usePureMask ?? false)
      setShowHudOverlay(hud.showHudOverlay ?? false)
      if (typeof ui.projectionScale === 'number') setProjectionScale(ui.projectionScale)
    })()
    return () => {
      cancelled = true
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
          if (hud.maskDisplayMode !== undefined || hud.showColor !== undefined) {
            setMaskDisplayMode(resolveMaskDisplayMode(hud))
          }
          if (hud.outlineMode !== undefined) setOutlineMode(hud.outlineMode)
          if (hud.maskOpacity !== undefined) setMaskOpacity(hud.maskOpacity)
          if (hud.registrationMode !== undefined) setRegistrationMode(hud.registrationMode)
          if (hud.blackScreen !== undefined) setBlackScreen(hud.blackScreen)
          if (hud.whiteScreen !== undefined) setWhiteScreen(hud.whiteScreen)
          if (hud.showDoneLayers !== undefined) setShowDoneLayers(hud.showDoneLayers)
          if (hud.showFinalPreview !== undefined) setShowFinalPreview(hud.showFinalPreview)
          if (hud.showOriginalImage !== undefined) setShowOriginalImage(hud.showOriginalImage)
          if (hud.usePureMask !== undefined) setUsePureMask(hud.usePureMask)
          if (hud.showHudOverlay !== undefined) setShowHudOverlay(hud.showHudOverlay)
        } catch (_) {}
      }
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [sessionId])

  useEffect(() => {
    if (!sessionId) return
    const hud = {
      crosshairs,
      grid,
      inverted,
      maskDisplayMode,
      outlineMode,
      maskOpacity,
      registrationMode,
      blackScreen,
      whiteScreen,
      showDoneLayers,
      showFinalPreview,
      showOriginalImage,
      usePureMask,
      showHudOverlay,
    }
    saveViewerHudState(sessionId, hud)
    void saveProjectState(sessionId, {
      currentLayer,
      doneLayers: Array.from(doneLayers),
      projectionScale,
      projectionHud: hud,
    })
  }, [
    sessionId,
    currentLayer,
    doneLayers,
    projectionScale,
    crosshairs,
    grid,
    inverted,
    maskDisplayMode,
    outlineMode,
    maskOpacity,
    registrationMode,
    blackScreen,
    whiteScreen,
    showDoneLayers,
    showFinalPreview,
    showOriginalImage,
    usePureMask,
    showHudOverlay,
  ])

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
    if (!sessionId) return
    const id = window.setInterval(() => {
      void fetchProjectState(sessionId).then((ui) => {
        if (typeof ui.currentLayer === 'number' && ui.currentLayer >= 0) {
          setCurrentLayerState((prev) => (prev === ui.currentLayer ? prev : ui.currentLayer!))
        }
        if (typeof ui.projectionScale === 'number') {
          const n = ui.projectionScale
          if (n >= 0.25 && n <= 2) setProjectionScale((prev) => (prev === n ? prev : n))
        }
        if (Array.isArray(ui.doneLayers)) {
          setDoneLayers((prev) => {
            const next = new Set(ui.doneLayers)
            if (prev.size === next.size && [...prev].every((x) => next.has(x))) return prev
            return next
          })
        }
      })
    }, 5000)
    return () => clearInterval(id)
  }, [sessionId])

  const saveDoneLayers = useCallback((_layers: Set<number>) => {
    /* persisted via saveProjectState effect */
  }, [])

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
    const assetVersion = sessionData.artifacts_version
    const checkLayer = async (layerIndex: number): Promise<boolean> => {
      const layer = sessionData.layers[layerIndex]
      if (!layer?.mask_url) return false
      const url = projectAssetUrl(
        usePureMask && layer.mask_pure_url ? layer.mask_pure_url : layer.mask_url,
        sessionData.artifacts_version
      )
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
        case 'i':
          setInverted((p) => {
            if (!p) setMaskDisplayMode('white')
            return !p
          })
          break
        case 'k':
          setMaskDisplayMode((mode) => {
            const next = cycleMaskDisplayMode(mode)
            if (next !== 'white') setInverted(false)
            return next
          })
          break
        case 'o':
          setOutlineMode((p) => (['off', 'thin', 'thick', 'glow'] as OutlineMode[])[((['off', 'thin', 'thick', 'glow'].indexOf(p)) + 1) % 4])
          break
        case '[': setMaskOpacity((p) => Math.max(40, p - 5)); break
        case ']': setMaskOpacity((p) => Math.min(100, p + 5)); break
        case 'r': setRegistrationMode((p) => !p); break
        case 'b': setBlackScreen((p) => !p); setWhiteScreen(false); break
        case 'w': setWhiteScreen((p) => !p); setBlackScreen(false); break
        case 'h':
          setShowHudOverlay((p) => !p)
          break
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
    if (maskDisplayMode === 'white' || !sessionData || currentLayer < 0 || currentLayer >= sessionData.layers.length) {
      setColoredMaskUrl(null)
      return
    }
    const layerData = sessionData.layers[currentLayer]
    if (!layerData || layerData.is_finished) {
      setColoredMaskUrl(null)
      return
    }
    if (!layerData.mask_url) {
      setColoredMaskUrl(null)
      return
    }

    const assetVersion = sessionData.artifacts_version
    const purePath =
      layerData.mask_pure_url ??
      `/api/projects/${sessionData.session_id}/artifacts/layer_${layerData.layer_index}_pure_mask.png`
    const expandedPath = layerData.mask_url
    let cancelled = false

    const fail = (message: string) => {
      if (!cancelled) {
        setColoredMaskUrl(null)
        setMaskLoadError(message)
      }
    }

    ;(async () => {
      try {
        if (maskDisplayMode === 'detail') {
          const dataUrl = await buildDetailMaskDataUrl(
            projectAssetUrl(expandedPath, assetVersion),
            projectAssetUrl(purePath, assetVersion)
          )
          if (!cancelled) {
            setColoredMaskUrl(dataUrl)
            setMaskLoadError(null)
          }
          return
        }

        let colorHex: string | null = null
        if (layerData.is_gradient && layerData.hex) colorHex = layerData.hex
        else {
          const pc = sessionData.palette.find((p) => p.index === layerData.palette_index)
          if (pc?.hex) colorHex = pc.hex
        }
        if (!colorHex) {
          fail('No palette color for layer')
          return
        }

        const maskPath = usePureMask ? purePath : expandedPath
        const img = await new Promise<HTMLImageElement>((resolve, reject) => {
          const el = new Image()
          el.crossOrigin = 'anonymous'
          el.onload = () => resolve(el)
          el.onerror = () => reject(new Error('mask load failed'))
          el.src = projectAssetUrl(maskPath, assetVersion)
        })
        const dataUrl = buildColorMaskDataUrl(img, colorHex)
        if (!cancelled) {
          setColoredMaskUrl(dataUrl)
          setMaskLoadError(null)
        }
      } catch {
        fail(
          maskDisplayMode === 'detail'
            ? 'Detail mask failed to load (need expanded + pure masks)'
            : usePureMask
              ? 'Pure mask failed to load'
              : 'Mask failed to load'
        )
      }
    })()

    return () => {
      cancelled = true
    }
  }, [maskDisplayMode, currentLayer, sessionData, usePureMask])

  useEffect(() => {
    setMaskLoadError(null)
  }, [currentLayer, usePureMask, maskDisplayMode])

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

  const assetVersion = sessionData.artifacts_version
  const originalImageUrl = sessionData.original_url
    ? projectAssetUrl(sessionData.original_url, assetVersion)
    : null
  const outlineUrl =
    currentLayerData.is_finished || outlineMode === 'off'
      ? null
      : projectAssetUrl(
          String(currentLayerData[`outline_${outlineMode}_url` as keyof Layer] ?? ''),
          assetVersion
        ) || null
  const finishedLayer = sessionData.layers.find((l) => l.is_finished)
  const finalPreviewUrl = finishedLayer
    ? projectAssetUrl(finishedLayer.finished_url || finishedLayer.mask_url, assetVersion)
    : sessionData.quantized_preview_url
      ? projectAssetUrl(sessionData.quantized_preview_url, assetVersion)
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

          {showHudOverlay && (
            <div
              className="fixed top-4 right-4 z-[55] max-w-sm rounded-lg bg-black/85 text-white p-3 text-sm shadow-lg border border-white/10 pointer-events-none"
              style={{ opacity: mouseActive ? 1 : 0.85 }}
            >
              <div className="font-semibold text-base mb-2 border-b border-white/20 pb-2">Projection HUD</div>
              <div className="space-y-1.5">
                <div>
                  {!currentLayerData
                    ? '—'
                    : currentLayerData.is_finished
                      ? 'Finished'
                      : `Layer ${currentLayer + 1} / ${sessionData.layers.length}`}
                </div>
                {!currentLayerData?.is_finished &&
                  (currentLayerData.is_gradient ? (
                    <div className="text-gray-300">
                      {(() => {
                        const stepNum =
                          (currentLayerData.gradient_step_index ?? 0) >= 0
                            ? (currentLayerData.gradient_step_index ?? 0) + 1
                            : 0
                        const src = (currentLayerData as Layer & { source_palette_indices?: number[] }).source_palette_indices
                        if (src?.length === 1) return `Gradient ${stepNum} → Palette ${src[0]}`
                        if (src?.length) return `Gradient ${stepNum} → Palettes ${src.join(', ')}`
                        return `Gradient step ${stepNum}`
                      })()}
                    </div>
                  ) : (
                    <div className="text-gray-300">Palette {currentLayerData.palette_index}</div>
                  ))}
                <div className="text-xs text-gray-400 pt-2 border-t border-white/15 leading-relaxed space-y-1">
                  {PROJECTION_SHORTCUTS_LINES.map((line) => (
                    <div key={line}>{line}</div>
                  ))}
                </div>
                <div className="text-xs text-gray-500">
                  Mask: {maskDisplayModeLabel(maskDisplayMode)}
                </div>
              </div>
            </div>
          )}

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
                  src={
                    showFinalPreview && finalPreviewUrl
                      ? finalPreviewUrl
                      : projectAssetUrl(
                          currentLayerData.finished_url || currentLayerData.mask_url,
                          assetVersion
                        )
                  }
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
                  {maskDisplayMode !== 'white' && coloredMaskUrl ? (
                    <img
                      src={coloredMaskUrl}
                      alt={`Layer ${currentLayer + 1}`}
                      className="absolute"
                      style={{
                        opacity: registrationMode ? 0 : maskOpacity / 100,
                        filter: inverted ? 'invert(1)' : 'none',
                        maxWidth: '100%',
                        maxHeight: '100%',
                        objectFit: 'contain',
                      }}
                    />
                  ) : (
                    <img
                      src={projectAssetUrl(
                        usePureMask
                          ? (currentLayerData.mask_pure_url ??
                              `/api/projects/${sessionData.session_id}/artifacts/layer_${currentLayerData.layer_index}_pure_mask.png`)
                          : currentLayerData.mask_url,
                        assetVersion
                      )}
                      alt={`Layer ${currentLayer + 1}`}
                      className="absolute"
                      crossOrigin="anonymous"
                      onLoad={() => setMaskLoadError(null)}
                      onError={() =>
                        setMaskLoadError(usePureMask ? 'Pure mask failed to load' : 'Mask failed to load')
                      }
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
