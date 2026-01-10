'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { API_BASE_URL } from '@/lib/config'

interface Layer {
  layer_index: number
  palette_index: number
  mask_url: string
  outline_thin_url: string
  outline_thick_url: string
  outline_glow_url: string
  is_finished?: boolean
  finished_url?: string
}

interface SessionData {
  session_id: string
  width: number
  height: number
  palette: Array<{ index: number; hex: string; coverage: number }>
  order: number[]
  layers: Layer[]
}

type OutlineMode = 'off' | 'thin' | 'thick' | 'glow'

export default function ProjectionViewer() {
  const params = useParams()
  const router = useRouter()
  const sessionId = params.sessionId as string
  const [sessionData, setSessionData] = useState<SessionData | null>(null)
  const [currentLayer, setCurrentLayer] = useState(0)
  const [crosshairs, setCrosshairs] = useState(true)
  const [grid, setGrid] = useState(false)
  const [inverted, setInverted] = useState(false)
  const [showColor, setShowColor] = useState(false)
  const [outlineMode, setOutlineMode] = useState<OutlineMode>('thin')
  const [maskOpacity, setMaskOpacity] = useState(85)
  const [registrationMode, setRegistrationMode] = useState(false)
  const [blackScreen, setBlackScreen] = useState(false)
  const [whiteScreen, setWhiteScreen] = useState(false)
  const [showHUD, setShowHUD] = useState(true)
  const [mouseActive, setMouseActive] = useState(true)
  const [doneLayers, setDoneLayers] = useState<Set<number>>(new Set())
  const [showDoneLayers, setShowDoneLayers] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const mouseTimerRef = useRef<NodeJS.Timeout>()

  // Load session data
  useEffect(() => {
    const stored = localStorage.getItem(`session_${sessionId}`)
    if (stored) {
      try {
        const data = JSON.parse(stored)
        setSessionData(data)
        const done = localStorage.getItem(`done_${sessionId}`)
        if (done) {
          setDoneLayers(new Set(JSON.parse(done)))
        }
      } catch (e) {
        console.error('Failed to load session data')
      }
    }
  }, [sessionId])

  // Load done layers from localStorage
  useEffect(() => {
    const done = localStorage.getItem(`done_${sessionId}`)
    if (done) {
      setDoneLayers(new Set(JSON.parse(done)))
    }
  }, [sessionId])

  // Save done layers to localStorage
  const saveDoneLayers = useCallback((layers: Set<number>) => {
    localStorage.setItem(`done_${sessionId}`, JSON.stringify(Array.from(layers)))
  }, [sessionId])

  // Navigation and done functions
  const navigateLayer = useCallback((direction: number) => {
    if (!sessionData) return
    let next = currentLayer + direction
    const maxLayer = sessionData.layers.length - 1

    // Skip done layers only if showDoneLayers is false (but don't skip the finished layer)
    if (!showDoneLayers) {
      while (next >= 0 && next <= maxLayer && doneLayers.has(next) && !sessionData.layers[next]?.is_finished) {
        next += direction
      }
    }

    if (next >= 0 && next <= maxLayer) {
      setCurrentLayer(next)
    }
  }, [sessionData, currentLayer, doneLayers, showDoneLayers])

  const toggleDone = useCallback(() => {
    // Don't allow marking the finished layer as done
    if (sessionData?.layers[currentLayer]?.is_finished) {
      return
    }
    const newDone = new Set(doneLayers)
    if (newDone.has(currentLayer)) {
      newDone.delete(currentLayer)
    } else {
      newDone.add(currentLayer)
    }
    setDoneLayers(newDone)
    saveDoneLayers(newDone)
  }, [currentLayer, doneLayers, saveDoneLayers, sessionData])

  // Mouse auto-hide
  useEffect(() => {
    const handleMouseMove = () => {
      setMouseActive(true)
      if (mouseTimerRef.current) {
        clearTimeout(mouseTimerRef.current)
      }
      mouseTimerRef.current = setTimeout(() => {
        setMouseActive(false)
      }, 2000)
    }

    window.addEventListener('mousemove', handleMouseMove)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      if (mouseTimerRef.current) {
        clearTimeout(mouseTimerRef.current)
      }
    }
  }, [])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return

      switch (e.key.toLowerCase()) {
        case 'c':
          setCrosshairs((prev) => !prev)
          break
        case 'g':
          setGrid((prev) => !prev)
          break
        case 'i':
          setInverted((prev) => {
            const newInverted = !prev
            // When enabling invert, turn off color mode
            if (newInverted) {
              setShowColor(false)
            }
            return newInverted
          })
          break
        case 'k':
          // K for color (Kolor) - toggle actual color display
          setShowColor((prev) => {
            const newShowColor = !prev
            // When enabling color, turn off invert
            if (newShowColor) {
              setInverted(false)
            }
            return newShowColor
          })
          break
        case 'o':
          setOutlineMode((prev) => {
            const modes: OutlineMode[] = ['off', 'thin', 'thick', 'glow']
            const idx = modes.indexOf(prev)
            return modes[(idx + 1) % modes.length]
          })
          break
        case '[':
          setMaskOpacity((prev) => Math.max(40, prev - 5))
          break
        case ']':
          setMaskOpacity((prev) => Math.min(100, prev + 5))
          break
        case 'r':
          setRegistrationMode((prev) => !prev)
          break
        case 'b':
          setBlackScreen((prev) => !prev)
          setWhiteScreen(false)
          break
        case 'w':
          setWhiteScreen((prev) => !prev)
          setBlackScreen(false)
          break
        case 'h':
          setShowHUD((prev) => !prev)
          break
        case 'arrowleft':
          navigateLayer(-1)
          break
        case 'arrowright':
          navigateLayer(1)
          break
        case ' ':
          e.preventDefault()
          navigateLayer(1)
          break
        case 'd':
          toggleDone()
          break
        case 's':
          // S for Show - toggle showing done layers
          setShowDoneLayers((prev) => !prev)
          break
        case 'escape':
        case 'Escape':
          // Save current session ID before navigating back
          localStorage.setItem('current_session_id', sessionId)
          router.back()
          break
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [navigateLayer, toggleDone, router, showDoneLayers])

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

  // Get the color for this layer
  const layerColor = sessionData.palette.find(p => p.index === currentLayerData.palette_index)

  const baseUrl = API_BASE_URL
  const outlineUrl =
    currentLayerData.is_finished || outlineMode === 'off'
      ? null
      : `${baseUrl}${currentLayerData[`outline_${outlineMode}_url` as keyof Layer]}`

  return (
    <div
      ref={containerRef}
      id="projection-viewer"
      className={`fixed inset-0 bg-black ${mouseActive ? 'show-cursor' : ''}`}
      style={{ cursor: mouseActive ? 'default' : 'none' }}
    >
      {blackScreen && (
        <div className="fixed inset-0 bg-black z-50" />
      )}
      {whiteScreen && (
        <div className="fixed inset-0 bg-white z-50" />
      )}

      {!blackScreen && !whiteScreen && (
        <>
          {/* Back button - only show when HUD is visible */}
          {showHUD && (
            <button
              onClick={() => {
                // Save current session ID before navigating back
                localStorage.setItem('current_session_id', sessionId)
                router.back()
              }}
              className="fixed top-4 left-4 z-50 px-4 py-2 bg-black bg-opacity-70 hover:bg-opacity-90 text-white rounded flex items-center gap-2 transition-opacity"
              style={{ opacity: mouseActive ? 1 : 0.3 }}
            >
              <span>←</span> Back
            </button>
          )}

          {/* Main canvas */}
          <div className="relative w-full h-full flex items-center justify-center">
            {/* Finished image or Mask image */}
            {currentLayerData.is_finished ? (
              <img
                src={`${baseUrl}${currentLayerData.finished_url || currentLayerData.mask_url}`}
                alt="Finished Image"
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
                {/* Mask image - with color or monochrome */}
                {showColor && layerColor ? (
                  <div
                    className="absolute"
                    style={{
                      opacity: registrationMode ? 0 : maskOpacity / 100,
                      maxWidth: '100%',
                      maxHeight: '100%',
                      width: '100%',
                      height: '100%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    {/* Colored div with mask applied - mask-image makes white areas show color */}
                    <div
                      className="absolute"
                      style={{
                        backgroundColor: layerColor.hex,
                        maskImage: `url(${baseUrl}${currentLayerData.mask_url})`,
                        WebkitMaskImage: `url(${baseUrl}${currentLayerData.mask_url})`,
                        maskSize: 'contain',
                        WebkitMaskSize: 'contain',
                        maskRepeat: 'no-repeat',
                        WebkitMaskRepeat: 'no-repeat',
                        maskPosition: 'center',
                        WebkitMaskPosition: 'center',
                        maxWidth: '100%',
                        maxHeight: '100%',
                        width: '100%',
                        height: '100%',
                      }}
                    />
                  </div>
                ) : (
                  <img
                    src={`${baseUrl}${currentLayerData.mask_url}`}
                    alt={`Layer ${currentLayer}`}
                    className="absolute"
                    style={{
                      opacity: registrationMode ? 0 : maskOpacity / 100,
                      filter: inverted ? 'invert(1)' : 'none',
                      maxWidth: '100%',
                      maxHeight: '100%',
                      objectFit: 'contain',
                    }}
                  />
                )}

                {/* Outline overlay */}
                {outlineUrl && (
                  <img
                    src={outlineUrl}
                    alt="Outline"
                    className="absolute pointer-events-none"
                    style={{
                      maxWidth: '100%',
                      maxHeight: '100%',
                      objectFit: 'contain',
                    }}
                  />
                )}
              </>
            )}

            {/* Corner crosshairs */}
            {crosshairs && (
              <svg className="absolute inset-0 w-full h-full pointer-events-none">
                <defs>
                  <style>{`
                    .crosshair { stroke: #888; stroke-width: 2; }
                  `}</style>
                </defs>
                {/* Top-left */}
                <line
                  x1="3%"
                  y1="3%"
                  x2="3%"
                  y2="8%"
                  className="crosshair"
                />
                <line
                  x1="3%"
                  y1="3%"
                  x2="8%"
                  y2="3%"
                  className="crosshair"
                />
                {/* Top-right */}
                <line
                  x1="97%"
                  y1="3%"
                  x2="97%"
                  y2="8%"
                  className="crosshair"
                />
                <line
                  x1="97%"
                  y1="3%"
                  x2="92%"
                  y2="3%"
                  className="crosshair"
                />
                {/* Bottom-left */}
                <line
                  x1="3%"
                  y1="97%"
                  x2="3%"
                  y2="92%"
                  className="crosshair"
                />
                <line
                  x1="3%"
                  y1="97%"
                  x2="8%"
                  y2="97%"
                  className="crosshair"
                />
                {/* Bottom-right */}
                <line
                  x1="97%"
                  y1="97%"
                  x2="97%"
                  y2="92%"
                  className="crosshair"
                />
                <line
                  x1="97%"
                  y1="97%"
                  x2="92%"
                  y2="97%"
                  className="crosshair"
                />
              </svg>
            )}

            {/* Grid */}
            {grid && (
              <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-30">
                <defs>
                  <style>{`
                    .grid-line { stroke: #444; stroke-width: 1; }
                  `}</style>
                </defs>
                {Array.from({ length: 20 }).map((_, i) => (
                  <g key={i}>
                    <line
                      x1={`${(i + 1) * 5}%`}
                      y1="0%"
                      x2={`${(i + 1) * 5}%`}
                      y2="100%"
                      className="grid-line"
                    />
                    <line
                      x1="0%"
                      y1={`${(i + 1) * 5}%`}
                      x2="100%"
                      y2={`${(i + 1) * 5}%`}
                      className="grid-line"
                    />
                  </g>
                ))}
              </svg>
            )}
          </div>

          {/* HUD */}
          {showHUD && (
            <div className="fixed bottom-4 left-4 right-4 bg-black bg-opacity-70 p-4 rounded text-white text-sm">
              <div className="flex flex-wrap gap-4">
                <div>
                  {currentLayerData.is_finished 
                    ? 'Finished Image' 
                    : `Layer: ${currentLayer + 1} / ${sessionData.layers.length}`}
                </div>
                {!currentLayerData.is_finished && (
                  <>
                    <div>Opacity: {maskOpacity}%</div>
                    <div>Outline: {outlineMode}</div>
                    <div>{showColor ? 'Color ON' : inverted ? 'Inverted' : 'Normal'}</div>
                  </>
                )}
                <div>{crosshairs ? 'Crosshairs ON' : 'Crosshairs OFF'}</div>
                <div>{grid ? 'Grid ON' : 'Grid OFF'}</div>
                <div>{registrationMode ? 'Registration ON' : 'Registration OFF'}</div>
                <div>{showDoneLayers ? 'Show Done: ON' : 'Show Done: OFF'}</div>
                {!currentLayerData.is_finished && (
                  <div>{doneLayers.has(currentLayer) ? '✓ Done' : ''}</div>
                )}
              </div>
              <div className="mt-2 text-xs text-gray-400">
                ← → / Space: Navigate | D: Toggle Done | C: Crosshairs | G: Grid | I: Invert | K: Color | O: Outline | [ ]: Opacity | R: Registration | B: Black | W: White | S: Show Done | H: HUD | Esc: Back
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

