'use client'

import { useState, useEffect, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import type { SessionData } from './types'
import { getProjectBySessionId, removeProject, saveProject, syncProjectsFromServer } from '@/lib/projects'
import { SessionResultsContent } from './SessionResultsContent'
import { ProjectionHUDControls } from './ProjectionHUDControls'
import { ProjectionToolsPanel } from './ProjectionToolsPanel'
import { projectAssetUrl } from '@/lib/projectAssets'
import { projectionPopupOpenFeatures } from '@/lib/projectionWindowBounds'
import { fetchProjectSession, fetchProjectState, saveProjectState } from '@/lib/projectSession'

export type PanelMode = 'layers' | 'projection' | 'all'

export interface ProjectionControlPanelProps {
  sessionId: string
  /** When true, used inside home page: no Back/Edit/Delete links */
  embedMode?: boolean
  /** When 'layers': only layers content (no Projection window). When 'projection': only Projection window + HUD. When 'all' or undefined: everything. */
  panelMode?: PanelMode
  /** Bump after generate in Image tab so this panel reloads session from server. */
  sessionRevision?: number
  /** When provided (e.g. in embed mode), called after project is deleted instead of navigating */
  onProjectDeleted?: () => void
}

export function ProjectionControlPanel({
  sessionId,
  embedMode = false,
  panelMode = 'all',
  sessionRevision = 0,
  onProjectDeleted,
}: ProjectionControlPanelProps) {
  const router = useRouter()
  const [sessionData, setSessionData] = useState<SessionData | null>(null)
  const [sessionLoaded, setSessionLoaded] = useState(false)
  const [currentLayer, setCurrentLayer] = useState(0)
  const [projectName, setProjectName] = useState<string | null>(null)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  useEffect(() => {
    if (typeof window === 'undefined' || !sessionId) return
    void (async () => {
      await syncProjectsFromServer()
      const p = getProjectBySessionId(sessionId)
      setProjectName(p?.name ?? null)
    })()
  }, [sessionId])

  useEffect(() => {
    if (typeof window === 'undefined' || !sessionId || !sessionData) return
    const proj = getProjectBySessionId(sessionId)
    if (!proj) return

    const nextWidth =
      typeof sessionData.canvas_width_cm === 'number' && sessionData.canvas_width_cm > 0
        ? sessionData.canvas_width_cm
        : proj.canvasWidthCm
    const nextHeight =
      typeof sessionData.canvas_height_cm === 'number' && sessionData.canvas_height_cm > 0
        ? sessionData.canvas_height_cm
        : proj.canvasHeightCm

    if (nextWidth === proj.canvasWidthCm && nextHeight === proj.canvasHeightCm) return
    saveProject({
      ...proj,
      canvasWidthCm: nextWidth,
      canvasHeightCm: nextHeight,
    })
  }, [sessionId, sessionData])

  useEffect(() => {
    if (!sessionId) return
    let cancelled = false
    setSessionLoaded(false)
    setSessionData(null)
    void (async () => {
      const [session, ui] = await Promise.all([
        fetchProjectSession(sessionId),
        fetchProjectState(sessionId),
      ])
      if (cancelled) return
      if (session) {
        setSessionData(session)
        const maxIdx = Math.max(0, session.layers.length - 1)
        const layer = typeof ui.currentLayer === 'number' ? ui.currentLayer : 0
        setCurrentLayer(Math.min(Math.max(0, layer), maxIdx))
      } else {
        setSessionData(null)
        setCurrentLayer(0)
      }
      setSessionLoaded(true)
    })()
    return () => {
      cancelled = true
    }
  }, [sessionId, sessionRevision])

  useEffect(() => {
    if (!sessionId || !sessionLoaded) return
    const id = window.setInterval(() => {
      void fetchProjectState(sessionId).then((ui) => {
        if (typeof ui.currentLayer === 'number' && ui.currentLayer >= 0) {
          setCurrentLayer((prev) => (prev === ui.currentLayer ? prev : ui.currentLayer!))
        }
      })
    }, 5000)
    return () => clearInterval(id)
  }, [sessionId, sessionLoaded])

  useEffect(() => {
    if (!sessionId || !sessionLoaded) return
    void saveProjectState(sessionId, { currentLayer })
  }, [sessionId, sessionLoaded, currentLayer])

  const openProjectionWindow = useCallback(() => {
    if (!sessionId || typeof window === 'undefined') return
    const url = `${window.location.origin}/project/${sessionId}/viewer`
    window.open(url, 'projection', projectionPopupOpenFeatures())
  }, [sessionId])

  const handleDeleteProject = useCallback(() => {
    if (!sessionId || typeof window === 'undefined') return
    removeProject(sessionId)
    if (localStorage.getItem('layerpainter_current_session_id') === sessionId) {
      localStorage.removeItem('layerpainter_current_session_id')
    }
    setShowDeleteConfirm(false)
    if (onProjectDeleted) {
      onProjectDeleted()
    } else {
      router.push('/')
    }
  }, [sessionId, router, onProjectDeleted])

  const openImageTab = useCallback(() => {
    if (!sessionId || typeof window === 'undefined') return
    if (embedMode && window.top !== window.self) {
      window.top!.location.href = `/?tab=image&session=${sessionId}`
    } else {
      router.push(`/upload?edit=${sessionId}&returnTo=home`)
    }
  }, [sessionId, embedMode, router])

  if (!sessionLoaded) {
    return (
      <div className="flex items-center justify-center py-12 text-gray-400">
        Loading session...
      </div>
    )
  }

  if (!sessionData) {
    return (
      <div className="space-y-6">
        {!embedMode && (
          <div className="flex items-center justify-between flex-wrap gap-4">
            <Link href="/" className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded">← Back to home</Link>
            <h1 className="text-xl font-bold">{projectName ? projectName : `Project: ${sessionId.slice(0, 8)}…`}</h1>
          </div>
        )}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-2">Generate Layers</h2>
          <p className="text-gray-400 text-sm mb-4">
            No layers generated yet. Upload an image and set options on the Image tab, then generate.
          </p>
          <button
            type="button"
            onClick={openImageTab}
            className="px-6 py-3 bg-blue-600 hover:bg-blue-500 rounded font-medium"
          >
            Open Image tab to generate
          </button>
        </div>
      </div>
    )
  }

  const currentLayerData = sessionData.layers[currentLayer]
  const showLayersContent = panelMode === 'all' || panelMode === 'layers'
  const showProjectionContent = panelMode === 'all' || panelMode === 'projection'

  return (
    <div className="space-y-6">
      {!embedMode && (
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded flex items-center gap-2"
            >
              ← Back to home
            </Link>
            <Link
              href={`/upload?edit=${sessionId}`}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded flex items-center gap-2"
            >
              Edit image & settings
            </Link>
            <h1 className="text-xl font-bold">
              {projectName ? projectName : `Project: ${sessionId.slice(0, 8)}…`}
            </h1>
            <button
              type="button"
              onClick={() => setShowDeleteConfirm(true)}
              className="ml-auto px-4 py-2 bg-red-900/80 hover:bg-red-800 text-red-200 rounded"
            >
              Delete project
            </button>
          </div>
        </div>
      )}

      {showDeleteConfirm && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full border border-gray-600">
            <h2 className="text-lg font-semibold mb-2">Delete project?</h2>
            <p className="text-gray-400 text-sm mb-6">
              This will remove &quot;{projectName || sessionId.slice(0, 8)}&quot; from your project list and clear its saved data. This cannot be undone.
            </p>
            <div className="flex justify-end gap-3">
              <button
                type="button"
                onClick={() => setShowDeleteConfirm(false)}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleDeleteProject}
                className="px-4 py-2 bg-red-600 hover:bg-red-500 rounded"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {showLayersContent && (
        <SessionResultsContent sessionId={sessionId} sessionData={sessionData} />
      )}

      {showProjectionContent && (
        <>
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-4">Projection window</h2>
            <p className="text-gray-400 text-sm mb-4">
              Open the projection in a separate window so you can move it to your second monitor or projector. Use the controls below to change layer, mask, opacity, and other options; they sync with the projection window.
            </p>
            <button
              type="button"
              onClick={openProjectionWindow}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-500 rounded font-medium"
            >
              Open projection window
            </button>
          </div>

          <ProjectionToolsPanel sessionId={sessionId} />

          <ProjectionHUDControls sessionId={sessionId} sessionData={sessionData} />

          {sessionData?.layers && (() => {
            const layer = sessionData.layers[currentLayer]
            if (!layer) return null
            const isGradient = layer.is_gradient ?? false
            let colorHex = '#000000'
            let paletteLabel = ''
            if (isGradient) {
              colorHex = layer.hex ?? '#808080'
              const stepNum = ((layer.gradient_step_index ?? 0) >= 0 ? (layer.gradient_step_index ?? 0) + 1 : 0)
              const src = (layer as { source_palette_indices?: number[] }).source_palette_indices
              paletteLabel = src?.length === 1 ? `Gradient ${stepNum} (→ Palette ${src[0]})` : src?.length ? `Gradient ${stepNum} (→ ${src.join(', ')})` : `Gradient ${stepNum}`
            } else {
              const color = sessionData.palette.find((p) => p.index === layer.palette_index)
              if (!color) return null
              colorHex = color.hex
              paletteLabel = `Palette ${layer.palette_index}`
            }
            return (
              <div className="mt-6 bg-gray-800 rounded-lg p-6">
                <h2 className="text-xl font-bold mb-3">Current layer</h2>
                <div className={`flex items-center gap-4 p-3 rounded ${isGradient ? 'bg-purple-900/30 border border-purple-700' : 'bg-gray-700'}`}>
                  <div className="text-lg font-mono font-semibold w-8">{layer.layer_index + 1}</div>
                  <div
                    className="w-12 h-12 rounded border border-gray-600 flex-shrink-0"
                    style={{ backgroundColor: colorHex }}
                    title={colorHex}
                  />
                  <img
                    src={projectAssetUrl(
                      layer.mask_pure_url ?? `/api/projects/${sessionId}/artifacts/layer_${layer.layer_index}_pure_mask.png`,
                      sessionData.artifacts_version
                    )}
                    alt={`Layer ${layer.layer_index + 1}`}
                    className="w-12 h-12 object-contain bg-gray-600 rounded flex-shrink-0"
                  />
                  <div className="text-sm text-gray-300 font-medium">{paletteLabel}</div>
                </div>
              </div>
            )
          })()}
        </>
      )}
    </div>
  )
}
