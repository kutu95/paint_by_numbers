'use client'

import { useEffect, useState } from 'react'
import type { Project } from '@/lib/projects'
import { projectAssetUrl } from '@/lib/projectAssets'

export type ProjectFileViewMode = 'list' | 'grid'

const VIEW_MODE_KEY = 'layerpainter_file_view'

function formatCanvasSize(p: Project): string {
  const w = p.canvasWidthCm
  const h = p.canvasHeightCm
  if (w > 0 && h > 0) return `${w}×${h} cm`
  return '—'
}

function projectTitle(p: Project): string {
  return p.name?.trim() || 'Untitled'
}

interface ProjectFileListProps {
  projects: Project[]
  currentSessionId: string | null
  onSelect: (sessionId: string) => void
}

export function ProjectFileList({ projects, currentSessionId, onSelect }: ProjectFileListProps) {
  const [viewMode, setViewMode] = useState<ProjectFileViewMode>('list')

  useEffect(() => {
    if (typeof window === 'undefined') return
    const stored = localStorage.getItem(VIEW_MODE_KEY)
    if (stored === 'list' || stored === 'grid') setViewMode(stored)
  }, [])

  const setMode = (mode: ProjectFileViewMode) => {
    setViewMode(mode)
    if (typeof window !== 'undefined') localStorage.setItem(VIEW_MODE_KEY, mode)
  }

  return (
    <div>
      <div className="flex items-center justify-between gap-3 mb-2">
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wide">Recent projects</h2>
        <div
          className="flex rounded-md border border-gray-600 overflow-hidden text-xs"
          role="group"
          aria-label="Project list view"
        >
          <button
            type="button"
            onClick={() => setMode('list')}
            className={`px-2.5 py-1 ${viewMode === 'list' ? 'bg-gray-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-white'}`}
          >
            List
          </button>
          <button
            type="button"
            onClick={() => setMode('grid')}
            className={`px-2.5 py-1 border-l border-gray-600 ${viewMode === 'grid' ? 'bg-gray-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-white'}`}
          >
            Thumbnails
          </button>
        </div>
      </div>

      {viewMode === 'list' ? (
        <ul className="rounded-lg border border-gray-700 overflow-hidden divide-y divide-gray-700/80">
          {projects.map((p) => {
            const selected = currentSessionId === p.sessionId
            const title = projectTitle(p)
            const size = formatCanvasSize(p)
            const fileName = p.imageFileName?.trim()
            return (
              <li key={p.sessionId}>
                <button
                  type="button"
                  onClick={() => onSelect(p.sessionId)}
                  title={[title, size, fileName].filter(Boolean).join(' · ')}
                  className={`w-full flex items-center gap-2 sm:gap-3 px-3 py-1.5 text-left text-sm transition-colors min-h-[2rem] ${
                    selected ? 'bg-gray-700' : 'bg-gray-800/60 hover:bg-gray-700/70'
                  }`}
                >
                  <span className="flex-1 min-w-0 truncate font-medium text-white">{title}</span>
                  <span className="shrink-0 text-xs text-gray-500 tabular-nums">{size}</span>
                  {fileName ? (
                    <span className="hidden xl:inline min-w-0 max-w-[10rem] 2xl:max-w-[14rem] truncate text-xs text-gray-600">
                      {fileName}
                    </span>
                  ) : null}
                </button>
              </li>
            )
          })}
        </ul>
      ) : (
        <ul className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2">
          {projects.map((p) => {
            const selected = currentSessionId === p.sessionId
            const title = projectTitle(p)
            const thumbSrc = p.thumbUrl ? projectAssetUrl(p.thumbUrl) : null
            return (
              <li key={p.sessionId}>
                <button
                  type="button"
                  onClick={() => onSelect(p.sessionId)}
                  title={title}
                  className={`w-full rounded-lg border overflow-hidden text-left transition-colors ${
                    selected ? 'border-gray-500 bg-gray-700' : 'border-gray-700 bg-gray-800/60 hover:border-gray-600 hover:bg-gray-700/70'
                  }`}
                >
                  <div className="aspect-[4/3] bg-black/40 flex items-center justify-center overflow-hidden">
                    {thumbSrc ? (
                      <img
                        src={thumbSrc}
                        alt=""
                        className="w-full h-full object-cover"
                        loading="lazy"
                      />
                    ) : (
                      <span className="text-xs text-gray-600 px-2 text-center">No preview</span>
                    )}
                  </div>
                  <div className="px-2 py-1.5 min-w-0">
                    <p className="text-xs font-medium text-white truncate">{title}</p>
                    <p className="text-[10px] text-gray-500 truncate tabular-nums">{formatCanvasSize(p)}</p>
                  </div>
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
