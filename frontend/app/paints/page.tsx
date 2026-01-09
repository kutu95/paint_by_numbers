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

export default function PaintsPage() {
  const router = useRouter()
  const [paints, setPaints] = useState<Paint[]>([])
  const [loading, setLoading] = useState(true)
  const [showAddForm, setShowAddForm] = useState(false)
  const [editingPaint, setEditingPaint] = useState<Paint | null>(null)
  const [formData, setFormData] = useState({ name: '', hex_approx: '#000000', notes: '' })

  useEffect(() => {
    loadPaints()
  }, [])

  const loadPaints = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/paint/library`)
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

      if (editingPaint) {
        const response = await fetch(`${API_BASE_URL}/api/paint/library/${editingPaint.id}`, {
          method: 'PUT',
          body: formDataObj,
        })
        if (!response.ok) throw new Error('Failed to update paint')
      } else {
        const response = await fetch(`${API_BASE_URL}/api/paint/library`, {
          method: 'POST',
          body: formDataObj,
        })
        if (!response.ok) throw new Error('Failed to add paint')
      }

      setShowAddForm(false)
      setEditingPaint(null)
      setFormData({ name: '', hex_approx: '#000000', notes: '' })
      loadPaints()
    } catch (error) {
      console.error('Error:', error)
      alert('Failed to save paint')
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
      const response = await fetch(`${API_BASE_URL}/api/paint/library/${paintId}`, {
        method: 'DELETE',
      })
      if (!response.ok) throw new Error('Failed to delete paint')
      loadPaints()
    } catch (error) {
      console.error('Error:', error)
      alert('Failed to delete paint')
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

