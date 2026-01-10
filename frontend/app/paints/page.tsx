'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { API_BASE_URL } from '@/lib/config'

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

export default function PaintsPage() {
  const router = useRouter()
  const [paints, setPaints] = useState<Paint[]>([])
  const [loading, setLoading] = useState(true)
  const [showAddForm, setShowAddForm] = useState(false)
  const [editingPaint, setEditingPaint] = useState<Paint | null>(null)
  const [formData, setFormData] = useState({ name: '', hex_approx: '#000000', notes: '' })
  const [libraryGroups, setLibraryGroups] = useState<LibraryGroup[]>([])
  const [selectedGroup, setSelectedGroup] = useState<string>('default')
  const [showCreateGroup, setShowCreateGroup] = useState(false)
  const [newGroupName, setNewGroupName] = useState('')
  const [renamingGroup, setRenamingGroup] = useState<string | null>(null)
  const [renameGroupName, setRenameGroupName] = useState('')

  useEffect(() => {
    loadLibraryGroups()
  }, [])

  useEffect(() => {
    if (selectedGroup) {
      loadPaints()
    }
  }, [selectedGroup])

  const loadLibraryGroups = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/paint/library/groups`)
      const data = await response.json()
      setLibraryGroups(data.groups || [])
      if (data.groups && data.groups.length > 0 && !selectedGroup) {
        setSelectedGroup(data.groups[0].group)
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

  const handleLoadMatisseLibrary = async () => {
    if (!confirm('Add Derivan Matisse paints to your library? Existing paints with the same name will be skipped.')) return

    const matissePaints = [
      { name: 'Titanium White', hex_approx: '#F5F5F5', notes: 'Series 1' },
      { name: 'Red Oxide', hex_approx: '#A0522D', notes: 'Series 1' },
      { name: 'Phthalo Blue', hex_approx: '#003D82', notes: 'Series 2' },
      { name: 'Carbon Black', hex_approx: '#1A1A1A', notes: 'Series 1' },
      { name: 'Yellow Oxide', hex_approx: '#DAA520', notes: 'Series 1' },
      { name: 'Australian Olive Green', hex_approx: '#6B8E23', notes: 'Series 2' },
    ]

    try {
      let added = 0
      let skipped = 0
      
      for (const paint of matissePaints) {
        const formDataObj = new FormData()
        formDataObj.append('name', paint.name)
        formDataObj.append('hex_approx', paint.hex_approx)
        formDataObj.append('notes', paint.notes)

        try {
          const response = await fetch(`${API_BASE_URL}/api/paint/library`, {
            method: 'POST',
            body: formDataObj,
          })
          
          if (response.ok) {
            added++
          } else if (response.status === 400) {
            // Paint already exists, skip it
            skipped++
            const errorData = await response.json().catch(() => ({}))
            console.log(`Paint ${paint.name} already exists, skipping`)
          } else {
            // Other error
            const errorData = await response.json().catch(() => ({}))
            console.error(`Failed to add ${paint.name}:`, errorData)
          }
        } catch (error) {
          // Network error
          console.error(`Network error adding ${paint.name}:`, error)
        }
      }
      
      if (added > 0) {
        alert(`Added ${added} paint(s) to library. ${skipped > 0 ? `${skipped} paint(s) were already in the library.` : ''} Remember to calibrate them for accurate recipe generation.`)
      } else if (skipped === matissePaints.length) {
        alert('All Matisse paints are already in your library.')
      } else {
        alert('Failed to add some paints. Check the console for details.')
      }
      
      loadPaints()
    } catch (error) {
      console.error('Error:', error)
      alert('Failed to add Matisse paints')
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
            <button
              onClick={handleLoadMatisseLibrary}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded"
            >
              Load Matisse Library
            </button>
          </div>
        </div>

        {/* Library Group Selection */}
        <div className="mb-6 p-4 bg-gray-800 rounded">
          <div className="flex items-center gap-4 flex-wrap">
            <label className="font-semibold">Library Group:</label>
            <select
              value={selectedGroup}
              onChange={(e) => setSelectedGroup(e.target.value)}
              className="px-3 py-2 bg-gray-700 rounded border border-gray-600"
            >
              {libraryGroups.map((group) => (
                <option key={group.group} value={group.group}>
                  {group.name} ({group.paint_count} paints, {group.calibrated_count} calibrated)
                </option>
              ))}
            </select>
            <button
              onClick={() => setShowCreateGroup(true)}
              className="px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm"
            >
              + New Group
            </button>
          </div>
        </div>

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

