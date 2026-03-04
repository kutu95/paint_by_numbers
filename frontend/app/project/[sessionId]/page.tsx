'use client'

import { useParams } from 'next/navigation'
import { ProjectionControlPanel } from './ProjectionControlPanel'

export default function ProjectControlPage() {
  const params = useParams()
  const sessionId = params.sessionId as string

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-4xl mx-auto">
        <ProjectionControlPanel sessionId={sessionId} embedMode={false} />
      </div>
    </div>
  )
}
