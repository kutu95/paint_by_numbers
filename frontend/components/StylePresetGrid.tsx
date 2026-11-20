'use client'

import { useEffect, useRef, useState } from 'react'
import { API_BASE_URL } from '@/lib/config'
import { IMAGE_STYLE_PRESETS, type ImageStylePreset } from '@/lib/imageStylePresets'

type TileStatus = 'pending' | 'ready' | 'error'

interface StyleTile {
  id: ImageStylePreset
  label: string
  description: string
  status: TileStatus
  url: string | null
  error: string | null
}

function emptyTiles(): StyleTile[] {
  return IMAGE_STYLE_PRESETS.map((preset) => ({
    id: preset.id,
    label: preset.label,
    description: preset.description,
    status: 'pending',
    url: null,
    error: null,
  }))
}

function jpegBase64ToObjectUrl(jpegBase64: string): string {
  const binary = atob(jpegBase64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return URL.createObjectURL(new Blob([bytes], { type: 'image/jpeg' }))
}

export function StylePresetGrid({
  currentPreset,
  buildFormData,
  canGenerate,
  generating,
  hasLayers,
  generateHint,
  onClose,
  onRegenerate,
}: {
  currentPreset: ImageStylePreset
  buildFormData: () => FormData
  canGenerate: boolean
  generating: boolean
  hasLayers: boolean
  generateHint: string | null
  onClose: () => void
  onRegenerate: (preset: ImageStylePreset) => void
}) {
  const [tiles, setTiles] = useState<StyleTile[]>(emptyTiles)
  const [selected, setSelected] = useState<ImageStylePreset>(currentPreset)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [running, setRunning] = useState(true)
  const buildFormDataRef = useRef(buildFormData)
  buildFormDataRef.current = buildFormData

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  useEffect(() => {
    const controller = new AbortController()
    const urls: string[] = []

    const applyLine = (line: string) => {
      if (controller.signal.aborted || !line) return
      const item = JSON.parse(line) as { id?: string; jpeg_base64?: string; error?: string }
      if (!item.id) return
      const id = item.id as ImageStylePreset
      if (item.jpeg_base64) {
        const url = jpegBase64ToObjectUrl(item.jpeg_base64)
        urls.push(url)
        setTiles((prev) =>
          prev.map((tile) =>
            tile.id === id ? { ...tile, status: 'ready', url, error: null } : tile
          )
        )
        return
      }
      setTiles((prev) =>
        prev.map((tile) =>
          tile.id === id
            ? { ...tile, status: 'error', url: null, error: item.error || 'Preview failed' }
            : tile
        )
      )
    }

    void (async () => {
      setTiles(emptyTiles())
      setRunning(true)
      setLoadError(null)
      try {
        const response = await fetch(`${API_BASE_URL}/api/preview/quantize-styles`, {
          method: 'POST',
          body: buildFormDataRef.current(),
          signal: controller.signal,
        })
        if (!response.ok) {
          let detail = `HTTP ${response.status}`
          try {
            const err = await response.json()
            detail = (err as { detail?: string }).detail || detail
          } catch {
            detail = (await response.text()) || detail
          }
          throw new Error(detail)
        }
        if (!response.body) throw new Error('Preview stream was empty')

        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        while (true) {
          const { value, done } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          let newline = buffer.indexOf('\n')
          while (newline >= 0) {
            const line = buffer.slice(0, newline).trim()
            buffer = buffer.slice(newline + 1)
            if (line) applyLine(line)
            newline = buffer.indexOf('\n')
          }
        }
        buffer += decoder.decode()
        const tail = buffer.trim()
        if (tail) applyLine(tail)
      } catch (error) {
        if (controller.signal.aborted) return
        setLoadError(error instanceof Error ? error.message : 'Could not build style previews')
      } finally {
        if (!controller.signal.aborted) {
          setRunning(false)
          setTiles((prev) =>
            prev.map((tile) =>
              tile.status === 'pending'
                ? { ...tile, status: 'error', error: 'Preview did not finish' }
                : tile
            )
          )
        }
      }
    })()

    return () => {
      controller.abort()
      for (const url of urls) URL.revokeObjectURL(url)
    }
    // Load once per open. The panel is mounted only while it is visible.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const selectedTile = tiles.find((tile) => tile.id === selected) ?? tiles[0]
  const readyCount = tiles.filter((tile) => tile.status === 'ready').length
  const actionLabel = hasLayers ? 'Regenerate layers with this style' : 'Generate layers with this style'

  return (
    <div
      className="fixed inset-0 z-[80] bg-black/75 p-3 sm:p-4 flex"
      role="presentation"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="style-preset-grid-title"
        className="flex flex-col w-full h-full min-h-0 rounded-xl border border-gray-700 bg-gray-900 shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4 px-5 py-4 border-b border-gray-700 shrink-0">
          <div>
            <h2 id="style-preset-grid-title" className="text-xl font-semibold text-white">
              Style presets
            </h2>
            <p className="text-sm text-gray-400 mt-1 max-w-3xl">
              Every preset on this image, using your current colour count, detail, vibrancy, and priority region.
              Select one, then regenerate the paint layers.
            </p>
            <p className="text-xs text-gray-500 mt-1">
              {running
                ? `Building previews… ${readyCount} of ${tiles.length} ready.`
                : `${readyCount} of ${tiles.length} previews ready.`}
              {' '}
              Comparison frames are a quicker reduced-resolution pass. Layer generation still uses your full output resolution.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-gray-400 hover:text-white text-2xl font-bold w-10 h-10 flex items-center justify-center rounded hover:bg-gray-800 shrink-0"
            aria-label="Close style presets"
          >
            ×
          </button>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto px-5 py-4">
          {loadError && (
            <p className="mb-4 text-sm text-amber-300 bg-amber-950/40 border border-amber-900/50 rounded px-3 py-2">
              {loadError}
            </p>
          )}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {tiles.map((tile) => {
              const isSelected = tile.id === selected
              const isCurrent = tile.id === currentPreset
              return (
                <button
                  key={tile.id}
                  type="button"
                  onClick={() => setSelected(tile.id)}
                  aria-pressed={isSelected}
                  className={`text-left rounded-lg border bg-gray-950 overflow-hidden transition-colors ${
                    isSelected
                      ? 'border-blue-400 ring-2 ring-blue-400'
                      : 'border-gray-700 hover:border-gray-500'
                  }`}
                >
                  <div className="relative h-64 sm:h-80 bg-black flex items-center justify-center">
                    {tile.status === 'ready' && tile.url && (
                      <img
                        src={tile.url}
                        alt={`${tile.label} preview`}
                        className="w-full h-full object-contain"
                      />
                    )}
                    {tile.status === 'pending' && (
                      <p className="text-sm text-gray-500 animate-pulse px-6 text-center">Building preview…</p>
                    )}
                    {tile.status === 'error' && (
                      <p className="text-sm text-amber-300 px-6 text-center">{tile.error}</p>
                    )}
                    <div className="absolute top-2 left-2 flex gap-2">
                      {isCurrent && (
                        <span className="text-[11px] uppercase tracking-wide px-2 py-0.5 rounded bg-gray-900/85 text-gray-200 border border-gray-600">
                          Current
                        </span>
                      )}
                      {isSelected && (
                        <span className="text-[11px] uppercase tracking-wide px-2 py-0.5 rounded bg-blue-600 text-white">
                          Selected
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="px-3 py-3">
                    <p className="font-medium text-gray-100">{tile.label}</p>
                    <p className="text-xs text-gray-500 mt-1 leading-relaxed">{tile.description}</p>
                  </div>
                </button>
              )
            })}
          </div>
        </div>

        <div className="shrink-0 border-t border-gray-700 px-5 py-4 flex flex-col sm:flex-row sm:items-center gap-3 bg-gray-900 rounded-b-xl">
          <div className="min-w-0 flex-1">
            <p className="text-sm font-medium text-white">{selectedTile?.label}</p>
            <p className="text-xs text-gray-500 mt-0.5">{selectedTile?.description}</p>
            {generateHint && <p className="text-xs text-amber-300 mt-1">{generateHint}</p>}
          </div>
          <button
            type="button"
            onClick={() => onRegenerate(selected)}
            disabled={!canGenerate || generating || !selectedTile}
            className="px-5 py-3 bg-blue-600 hover:bg-blue-700 rounded disabled:opacity-50 disabled:cursor-not-allowed whitespace-normal text-sm font-medium shrink-0"
          >
            {generating ? 'Processing...' : actionLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
