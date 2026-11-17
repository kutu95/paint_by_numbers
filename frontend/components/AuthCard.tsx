import { ReactNode } from 'react'

export function AuthCard({ title, subtitle, children }: { title: string; subtitle?: string; children: ReactNode }) {
  return (
    <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center px-4">
      <div className="w-full max-w-md rounded-lg bg-gray-800 p-6">
        <h1 className="text-2xl font-bold">{title}</h1>
        {subtitle ? <p className="mt-2 text-sm text-gray-400">{subtitle}</p> : null}
        <div className="mt-6">{children}</div>
      </div>
    </div>
  )
}
