'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { API_BASE_URL } from '@/lib/config'

const CHART_WIDTH = 420
const CHART_HEIGHT = 180
const PAD = { left: 36, right: 12, top: 12, bottom: 28 }

function CalibrationResultsView({ data }: { data: CalibrationData }) {
  const samples = [...(data.samples || [])].sort((a, b) => a.ratio - b.ratio)
  if (samples.length === 0) return <p className="text-gray-400">No samples.</p>

  const ratios = samples.map((s) => s.ratio)
  const L = samples.map((s) => s.lab[0])
  const a = samples.map((s) => s.lab[1])
  const b = samples.map((s) => s.lab[2])
  const minRatio = Math.min(...ratios)
  const maxRatio = Math.max(...ratios)
  const minL = Math.min(0, ...L)
  const maxL = Math.max(100, ...L)
  const abExtent = Math.max(1, ...a.map(Math.abs), ...b.map(Math.abs), 50)
  const minAb = -abExtent
  const maxAb = abExtent

  // X: logarithmic scale, reversed (100% left, most diluted right); equal spacing per dilution step
  const logMin = Math.log2(Math.max(minRatio, 1e-6))
  const logMax = Math.log2(maxRatio)
  const logRange = logMax - logMin || 1
  const innerW = CHART_WIDTH - PAD.left - PAD.right
  const toX = (ratio: number) => {
    const t = (logMax - Math.log2(Math.max(ratio, 1e-6))) / logRange
    return PAD.left + t * innerW
  }
  const toYL = (v: number) =>
    PAD.top + (1 - (v - minL) / (maxL - minL || 1)) * (CHART_HEIGHT - PAD.top - PAD.bottom)
  const toYAb = (v: number) =>
    PAD.top + (1 - (v - minAb) / (maxAb - minAb || 1)) * (CHART_HEIGHT - PAD.top - PAD.bottom)

  const pathL = samples.map((s, i) => `${i === 0 ? 'M' : 'L'} ${toX(s.ratio)} ${toYL(s.lab[0])}`).join(' ')
  const pathA = samples.map((s, i) => `${i === 0 ? 'M' : 'L'} ${toX(s.ratio)} ${toYAb(s.lab[1])}`).join(' ')
  const pathB = samples.map((s, i) => `${i === 0 ? 'M' : 'L'} ${toX(s.ratio)} ${toYAb(s.lab[2])}`).join(' ')

  // Red / green from a* (red = +a*, green = -a*); yellow / blue from b* (yellow = +b*, blue = -b*)
  const redChroma = a.map((v) => Math.max(0, v))
  const greenChroma = a.map((v) => Math.max(0, -v))
  const yellowChroma = b.map((v) => Math.max(0, v))
  const blueChroma = b.map((v) => Math.max(0, -v))
  const maxChroma = Math.max(1, ...redChroma, ...greenChroma, ...yellowChroma, ...blueChroma)
  const toYChroma = (v: number) =>
    PAD.top + (1 - v / maxChroma) * (CHART_HEIGHT - PAD.top - PAD.bottom)
  const pathRed = samples.map((s, i) => `${i === 0 ? 'M' : 'L'} ${toX(s.ratio)} ${toYChroma(redChroma[i])}`).join(' ')
  const pathGreen = samples.map((s, i) => `${i === 0 ? 'M' : 'L'} ${toX(s.ratio)} ${toYChroma(greenChroma[i])}`).join(' ')
  const pathYellow = samples.map((s, i) => `${i === 0 ? 'M' : 'L'} ${toX(s.ratio)} ${toYChroma(yellowChroma[i])}`).join(' ')
  const pathBlue = samples.map((s, i) => `${i === 0 ? 'M' : 'L'} ${toX(s.ratio)} ${toYChroma(blueChroma[i])}`).join(' ')

  const rgbToHex = (r: number, g: number, blue: number) =>
    '#' + [r, g, blue].map((x) => Math.max(0, Math.min(255, Math.round(x))).toString(16).padStart(2, '0')).join('')

  return (
    <div className="space-y-4">
      {data.created_at && (
        <p className="text-sm text-gray-400">
          Calibrated: {new Date(data.created_at).toLocaleString()}
        </p>
      )}
      <div>
        <h4 className="text-sm font-medium text-gray-300 mb-2">Samples ({samples.length})</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-400 border-b border-gray-600">
                <th className="py-1 pr-3">Ratio</th>
                <th className="py-1 pr-2">Color</th>
                <th className="py-1 pr-2">L</th>
                <th className="py-1 pr-2">a*</th>
                <th className="py-1">b*</th>
              </tr>
            </thead>
            <tbody>
              {samples.map((s, i) => (
                <tr key={i} className="border-b border-gray-700">
                  <td className="py-1.5 pr-3">{(s.ratio * 100).toFixed(1)}%</td>
                  <td className="py-1.5 pr-2">
                    <span
                      className="inline-block w-6 h-6 rounded border border-gray-600"
                      style={{ backgroundColor: rgbToHex(s.rgb[0], s.rgb[1], s.rgb[2]) }}
                      title={rgbToHex(s.rgb[0], s.rgb[1], s.rgb[2])}
                    />
                  </td>
                  <td className="py-1.5 pr-2">{s.lab[0].toFixed(1)}</td>
                  <td className="py-1.5 pr-2">{s.lab[1].toFixed(1)}</td>
                  <td className="py-1.5">{s.lab[2].toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div>
        <h4 className="text-sm font-medium text-gray-300 mb-1">Luminance (L) vs dilution</h4>
        <p className="text-xs text-gray-500 mb-1">L: 0 (dark) → 100 (light)</p>
        <svg width={CHART_WIDTH} height={CHART_HEIGHT} className="overflow-visible">
          <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={CHART_HEIGHT - PAD.bottom} stroke="#4b5563" strokeWidth="1" />
          <line x1={PAD.left} y1={CHART_HEIGHT - PAD.bottom} x2={CHART_WIDTH - PAD.right} y2={CHART_HEIGHT - PAD.bottom} stroke="#4b5563" strokeWidth="1" />
          <text x={PAD.left - 8} y={PAD.top + 4} className="fill-gray-500 text-[10px]">100</text>
          <text x={PAD.left - 8} y={CHART_HEIGHT - PAD.bottom + 4} className="fill-gray-500 text-[10px]">0</text>
          {samples.map((s, i) => (
            <g key={i}>
              <line x1={toX(s.ratio)} y1={CHART_HEIGHT - PAD.bottom} x2={toX(s.ratio)} y2={CHART_HEIGHT - PAD.bottom + 4} stroke="#4b5563" strokeWidth="1" />
              <text x={toX(s.ratio)} y={CHART_HEIGHT - 4} textAnchor="middle" className="fill-gray-500 text-[10px]">{(s.ratio * 100).toFixed(s.ratio >= 0.1 ? 0 : 1)}%</text>
            </g>
          ))}
          <path d={pathL} fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>

      <div>
        <h4 className="text-sm font-medium text-gray-300 mb-1">Chroma (a*, b*) vs dilution</h4>
        <p className="text-xs text-gray-500 mb-1">a* (green–red), b* (blue–yellow)</p>
        <svg width={CHART_WIDTH} height={CHART_HEIGHT} className="overflow-visible">
          <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={CHART_HEIGHT - PAD.bottom} stroke="#4b5563" strokeWidth="1" />
          <line x1={PAD.left} y1={CHART_HEIGHT - PAD.bottom} x2={CHART_WIDTH - PAD.right} y2={CHART_HEIGHT - PAD.bottom} stroke="#4b5563" strokeWidth="1" />
          <text x={PAD.left - 8} y={PAD.top + 4} className="fill-gray-500 text-[10px]">{maxAb}</text>
          <text x={PAD.left - 8} y={CHART_HEIGHT - PAD.bottom + 4} className="fill-gray-500 text-[10px]">{minAb}</text>
          {samples.map((s, i) => (
            <g key={i}>
              <line x1={toX(s.ratio)} y1={CHART_HEIGHT - PAD.bottom} x2={toX(s.ratio)} y2={CHART_HEIGHT - PAD.bottom + 4} stroke="#4b5563" strokeWidth="1" />
              <text x={toX(s.ratio)} y={CHART_HEIGHT - 4} textAnchor="middle" className="fill-gray-500 text-[10px]">{(s.ratio * 100).toFixed(s.ratio >= 0.1 ? 0 : 1)}%</text>
            </g>
          ))}
          <path d={pathA} fill="none" stroke="#ef4444" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <path d={pathB} fill="none" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <text x={CHART_WIDTH - 80} y={PAD.top + 10} className="fill-[#ef4444] text-[10px]">a*</text>
          <text x={CHART_WIDTH - 50} y={PAD.top + 10} className="fill-[#3b82f6] text-[10px]">b*</text>
        </svg>
      </div>

      <div>
        <h4 className="text-sm font-medium text-gray-300 mb-1">Red / Green / Blue / Yellow chroma vs dilution</h4>
        <p className="text-xs text-gray-500 mb-1">From Lab: red = +a*, green = −a*, yellow = +b*, blue = −b*</p>
        <svg width={CHART_WIDTH} height={CHART_HEIGHT} className="overflow-visible">
          <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={CHART_HEIGHT - PAD.bottom} stroke="#4b5563" strokeWidth="1" />
          <line x1={PAD.left} y1={CHART_HEIGHT - PAD.bottom} x2={CHART_WIDTH - PAD.right} y2={CHART_HEIGHT - PAD.bottom} stroke="#4b5563" strokeWidth="1" />
          <text x={PAD.left - 8} y={PAD.top + 4} className="fill-gray-500 text-[10px]">{maxChroma.toFixed(0)}</text>
          <text x={PAD.left - 8} y={CHART_HEIGHT - PAD.bottom + 4} className="fill-gray-500 text-[10px]">0</text>
          {samples.map((s, i) => (
            <g key={i}>
              <line x1={toX(s.ratio)} y1={CHART_HEIGHT - PAD.bottom} x2={toX(s.ratio)} y2={CHART_HEIGHT - PAD.bottom + 4} stroke="#4b5563" strokeWidth="1" />
              <text x={toX(s.ratio)} y={CHART_HEIGHT - 4} textAnchor="middle" className="fill-gray-500 text-[10px]">{(s.ratio * 100).toFixed(s.ratio >= 0.1 ? 0 : 1)}%</text>
            </g>
          ))}
          <path d={pathRed} fill="none" stroke="#ef4444" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <path d={pathGreen} fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <path d={pathYellow} fill="none" stroke="#eab308" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <path d={pathBlue} fill="none" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <text x={CHART_WIDTH - 120} y={PAD.top + 8} className="fill-[#ef4444] text-[10px]">red</text>
          <text x={CHART_WIDTH - 95} y={PAD.top + 8} className="fill-[#22c55e] text-[10px]">green</text>
          <text x={CHART_WIDTH - 68} y={PAD.top + 8} className="fill-[#eab308] text-[10px]">yellow</text>
          <text x={CHART_WIDTH - 42} y={PAD.top + 8} className="fill-[#3b82f6] text-[10px]">blue</text>
        </svg>
      </div>
    </div>
  )
}

interface Paint {
  id: string
  name: string
  type: string
  hex_approx: string
  notes: string
}

interface LibraryGroup {
  group: string
  paint_count: number
  calibrated_count: number
  name: string
}

interface CalibrationSample {
  ratio: number
  rgb: number[]
  lab: number[]
}

interface CalibrationData {
  paint_id: string
  ratios: number[]
  samples: CalibrationSample[]
  reference_strip?: Record<string, { rgb: number[]; lab: number[] }>
  created_at?: string
  notes?: string
}

export default function PaintsPage() {
  const router = useRouter()
  const [paints, setPaints] = useState<Paint[]>([])
  const [loading, setLoading] = useState(true)
  const [showAddForm, setShowAddForm] = useState(false)
  const [editingPaint, setEditingPaint] = useState<Paint | null>(null)
  const [formData, setFormData] = useState({ name: '', hex_approx: '#000000', notes: '' })
  const [libraryGroups, setLibraryGroups] = useState<LibraryGroup[]>([])
  const [selectedGroup, setSelectedGroup] = useState<string>(() => {
    // Load last selected group from localStorage
    if (typeof window !== 'undefined') {
      return localStorage.getItem('lastSelectedPaintLibrary') || 'default'
    }
    return 'default'
  })
  const [showCreateGroup, setShowCreateGroup] = useState(false)
  const [newGroupName, setNewGroupName] = useState('')
  const [renamingGroup, setRenamingGroup] = useState<string | null>(null)
  const [renameGroupName, setRenameGroupName] = useState('')
  const [calibrationPanelOpen, setCalibrationPanelOpen] = useState(false)
  const [calibrationData, setCalibrationData] = useState<CalibrationData | null>(null)
  const [calibrationLoading, setCalibrationLoading] = useState(false)
  const [calibrationError, setCalibrationError] = useState<string | null>(null)

  useEffect(() => {
    loadLibraryGroups()
  }, [])

  useEffect(() => {
    if (selectedGroup) {
      // Save selected group to localStorage
      if (typeof window !== 'undefined') {
        localStorage.setItem('lastSelectedPaintLibrary', selectedGroup)
      }
      loadPaints()
    }
  }, [selectedGroup])

  useEffect(() => {
    if (!editingPaint) {
      setCalibrationPanelOpen(false)
      setCalibrationData(null)
      setCalibrationError(null)
      return
    }
    if (!calibrationPanelOpen) return
    setCalibrationLoading(true)
    setCalibrationError(null)
    fetch(`${API_BASE_URL}/api/paint/calibration/${editingPaint.id}`)
      .then((res) => {
        if (!res.ok) {
          if (res.status === 404) throw new Error('No calibration data')
          throw new Error(res.statusText)
        }
        return res.json()
      })
      .then((data: CalibrationData) => setCalibrationData(data))
      .catch((err: Error) => setCalibrationError(err.message))
      .finally(() => setCalibrationLoading(false))
  }, [editingPaint, calibrationPanelOpen])

  const loadLibraryGroups = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/paint/library/groups`)
      const data = await response.json()
      setLibraryGroups(data.groups || [])
      
      // If we have a saved group, verify it still exists, otherwise use first available
      if (data.groups && data.groups.length > 0) {
        const savedGroup = typeof window !== 'undefined' 
          ? localStorage.getItem('lastSelectedPaintLibrary') 
          : null
        const groupExists = savedGroup && data.groups.some((g: LibraryGroup) => g.group === savedGroup)
        
        if (groupExists && savedGroup) {
          setSelectedGroup(savedGroup)
        } else if (!selectedGroup || !data.groups.some((g: LibraryGroup) => g.group === selectedGroup)) {
          // Use first group if current selection doesn't exist
          setSelectedGroup(data.groups[0].group)
        }
      }
    } catch (error) {
      console.error('Failed to load library groups:', error)
    }
  }

  const loadPaints = async () => {
    if (!selectedGroup) return
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/paint/library?group=${selectedGroup}`)
      const data = await response.json()
      setPaints(data.paints || [])
    } catch (error) {
      console.error('Failed to load paints:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      const formDataObj = new FormData()
      formDataObj.append('name', formData.name)
      formDataObj.append('hex_approx', formData.hex_approx)
      formDataObj.append('notes', formData.notes)
      formDataObj.append('group', selectedGroup)

      if (editingPaint) {
        const response = await fetch(`${API_BASE_URL}/api/paint/library/${editingPaint.id}`, {
          method: 'PUT',
          body: formDataObj,
        })
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}))
          throw new Error(errorData.detail || 'Failed to update paint')
        }
        const updatedPaint: Paint = await response.json()
        setPaints((prev) => prev.map((p) => (p.id === updatedPaint.id ? { ...p, ...updatedPaint } : p)))
      } else {
        const response = await fetch(`${API_BASE_URL}/api/paint/library`, {
          method: 'POST',
          body: formDataObj,
        })
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}))
          const errorMessage = errorData.detail || 'Failed to add paint'
          if (response.status === 400 && errorMessage.includes('already exists')) {
            alert(`A paint with the name "${formData.name}" already exists in this library. Please use a different name or edit the existing paint.`)
          } else {
            throw new Error(errorMessage)
          }
          return
        }
      }

      setShowAddForm(false)
      setEditingPaint(null)
      setFormData({ name: '', hex_approx: '#000000', notes: '' })
      loadPaints()
    } catch (error) {
      console.error('Error:', error)
      const errorMessage = error instanceof Error ? error.message : 'Failed to save paint'
      alert(errorMessage)
    }
  }

  const handleEdit = (paint: Paint) => {
    setEditingPaint(paint)
    setFormData({ name: paint.name, hex_approx: paint.hex_approx, notes: paint.notes })
    setShowAddForm(true)
  }

  const handleDelete = async (paintId: string) => {
    if (!confirm('Delete this paint? This will also delete its calibration data.')) return

    try {
      const response = await fetch(`${API_BASE_URL}/api/paint/library/${paintId}?group=${selectedGroup}`, {
        method: 'DELETE',
      })
      if (!response.ok) throw new Error('Failed to delete paint')
      loadPaints()
      loadLibraryGroups() // Refresh group info
    } catch (error) {
      console.error('Error:', error)
      alert('Failed to delete paint')
    }
  }

  const handleCreateGroup = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newGroupName.trim()) return

    try {
      const formData = new FormData()
      formData.append('name', newGroupName.trim())

      const response = await fetch(`${API_BASE_URL}/api/paint/library/groups`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to create library group')
      }

      setNewGroupName('')
      setShowCreateGroup(false)
      loadLibraryGroups()
      // Switch to the new group
      const data = await response.json()
      setSelectedGroup(data.group)
    } catch (error) {
      console.error('Error:', error)
      alert(error instanceof Error ? error.message : 'Failed to create library group')
    }
  }

  const handleRenameGroup = async (group: string, currentName: string) => {
    setRenamingGroup(group)
    setRenameGroupName(currentName)
  }

  const handleRenameGroupSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!renamingGroup || !renameGroupName.trim()) return

    try {
      const formData = new FormData()
      formData.append('name', renameGroupName.trim())

      const response = await fetch(`${API_BASE_URL}/api/paint/library/groups/${renamingGroup}`, {
        method: 'PUT',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to rename library group')
      }

      setRenamingGroup(null)
      setRenameGroupName('')
      loadLibraryGroups()
    } catch (error) {
      console.error('Error:', error)
      alert(error instanceof Error ? error.message : 'Failed to rename library group')
    }
  }


  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 text-white p-8">
        <div className="max-w-6xl mx-auto">Loading...</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-4xl font-bold">Paint Library</h1>
          <div className="flex gap-4">
            <button
              onClick={() => router.push('/')}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
            >
              ← Back to Home
            </button>
            <button
              onClick={() => {
                setShowAddForm(true)
                setEditingPaint(null)
                setFormData({ name: '', hex_approx: '#000000', notes: '' })
              }}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded"
            >
              + Add Paint
            </button>
          </div>
        </div>

        {/* Library Group Selection */}
        <div className="mb-6 p-4 bg-gray-800 rounded">
          <div className="flex items-center gap-4 flex-wrap">
            <label className="font-semibold">Library Group:</label>
            <select
              value={selectedGroup}
              onChange={(e) => {
                const newGroup = e.target.value
                setSelectedGroup(newGroup)
                // Save to localStorage immediately
                if (typeof window !== 'undefined') {
                  localStorage.setItem('lastSelectedPaintLibrary', newGroup)
                }
              }}
              className="px-3 py-2 bg-gray-700 rounded border border-gray-600"
            >
              {libraryGroups.map((group) => (
                <option key={group.group} value={group.group}>
                  {group.name} ({group.paint_count} paints, {group.calibrated_count} calibrated)
                </option>
              ))}
            </select>
            <button
              onClick={() => {
                const currentGroup = libraryGroups.find(g => g.group === selectedGroup)
                if (currentGroup) {
                  handleRenameGroup(selectedGroup, currentGroup.name)
                }
              }}
              className="px-3 py-2 bg-gray-600 hover:bg-gray-500 rounded text-sm"
            >
              Rename Group
            </button>
            <button
              onClick={() => setShowCreateGroup(true)}
              className="px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm"
            >
              + New Group
            </button>
          </div>
        </div>

        {/* Rename Group Form */}
        {renamingGroup && (
          <div className="mb-6 p-4 bg-gray-800 rounded">
            <h3 className="font-bold mb-3">Rename Library Group</h3>
            <form onSubmit={handleRenameGroupSubmit} className="flex gap-3">
              <input
                type="text"
                value={renameGroupName}
                onChange={(e) => setRenameGroupName(e.target.value)}
                placeholder="New library name"
                className="flex-1 px-3 py-2 bg-gray-700 rounded border border-gray-600"
                required
                autoFocus
              />
              <button
                type="submit"
                className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded"
              >
                Rename
              </button>
              <button
                type="button"
                onClick={() => {
                  setRenamingGroup(null)
                  setRenameGroupName('')
                }}
                className="px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded"
              >
                Cancel
              </button>
            </form>
          </div>
        )}

        {/* Create New Group Form */}
        {showCreateGroup && (
          <div className="mb-6 p-4 bg-gray-800 rounded">
            <h3 className="font-bold mb-3">Create New Library Group</h3>
            <form onSubmit={handleCreateGroup} className="flex gap-3">
              <input
                type="text"
                value={newGroupName}
                onChange={(e) => setNewGroupName(e.target.value)}
                placeholder="Library name (e.g., Matisse, Dulux)"
                className="flex-1 px-3 py-2 bg-gray-700 rounded border border-gray-600"
                required
              />
              <button
                type="submit"
                className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded"
              >
                Create
              </button>
              <button
                type="button"
                onClick={() => {
                  setShowCreateGroup(false)
                  setNewGroupName('')
                }}
                className="px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded"
              >
                Cancel
              </button>
            </form>
          </div>
        )}

        {showAddForm && (
          <div className="mb-6 p-6 bg-gray-800 rounded">
            <h2 className="text-2xl font-bold mb-4">
              {editingPaint ? 'Edit Paint' : 'Add New Paint'}
            </h2>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block mb-2">Paint Name</label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  required
                  className="w-full px-3 py-2 bg-gray-700 rounded text-white"
                />
              </div>
              <div>
                <label className="block mb-2">Approximate Color (Hex)</label>
                <div className="flex gap-2">
                  <input
                    type="color"
                    value={formData.hex_approx}
                    onChange={(e) => setFormData({ ...formData, hex_approx: e.target.value })}
                    className="h-10 w-20"
                  />
                  <input
                    type="text"
                    value={formData.hex_approx}
                    onChange={(e) => setFormData({ ...formData, hex_approx: e.target.value })}
                    required
                    className="flex-1 px-3 py-2 bg-gray-700 rounded text-white"
                  />
                </div>
              </div>
              <div>
                <label className="block mb-2">Notes (optional)</label>
                <textarea
                  value={formData.notes}
                  onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-700 rounded text-white"
                  rows={3}
                />
              </div>

              {editingPaint && (
                <div className="border border-gray-600 rounded overflow-hidden">
                  <button
                    type="button"
                    onClick={() => setCalibrationPanelOpen((o) => !o)}
                    className="w-full px-4 py-3 flex items-center justify-between bg-gray-750 hover:bg-gray-700 text-left"
                  >
                    <span className="font-medium">Calibration results</span>
                    <span className="text-gray-400">{calibrationPanelOpen ? '▼' : '▶'}</span>
                  </button>
                  {calibrationPanelOpen && (
                    <div className="p-4 bg-gray-800/80 border-t border-gray-600 space-y-4">
                      {calibrationLoading && (
                        <p className="text-gray-400">Loading calibration…</p>
                      )}
                      {calibrationError && (
                        <p className="text-amber-400">{calibrationError}</p>
                      )}
                      {calibrationData && !calibrationLoading && (
                        <CalibrationResultsView data={calibrationData} />
                      )}
                    </div>
                  )}
                </div>
              )}

              <div className="flex gap-2">
                <button
                  type="submit"
                  className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded"
                >
                  {editingPaint ? 'Update' : 'Add'} Paint
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setShowAddForm(false)
                    setEditingPaint(null)
                    setFormData({ name: '', hex_approx: '#000000', notes: '' })
                  }}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {paints.map((paint) => (
            <div key={paint.id} className="p-4 bg-gray-800 rounded">
              <div className="flex items-center gap-3 mb-3">
                <div
                  className="w-16 h-16 rounded border border-gray-600"
                  style={{ backgroundColor: paint.hex_approx }}
                />
                <div className="flex-1">
                  <h3 className="text-lg font-bold">{paint.name}</h3>
                  <div className="text-sm text-gray-400">{paint.hex_approx}</div>
                </div>
              </div>
              {paint.notes && (
                <p className="text-sm text-gray-300 mb-3">{paint.notes}</p>
              )}
              <div className="flex gap-2">
                <button
                  onClick={() => router.push(`/paints/calibrate/${paint.id}`)}
                  className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm"
                >
                  Calibrate
                </button>
                <button
                  onClick={() => handleEdit(paint)}
                  className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm"
                >
                  Edit
                </button>
                <button
                  onClick={() => handleDelete(paint.id)}
                  className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>

        {paints.length === 0 && !showAddForm && (
          <div className="text-center py-12 text-gray-400">
            No paints in library. Click "Add Paint" to get started.
          </div>
        )}
      </div>
    </div>
  )
}

