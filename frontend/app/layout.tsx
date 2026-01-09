import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'LayerPainter - Projection Paint Layers',
  description: 'Projection-first paint-by-numbers layers app',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}

