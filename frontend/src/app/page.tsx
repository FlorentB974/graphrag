'use client'

import ChatInterface from '@/components/Chat/ChatInterface'
import Sidebar from '@/components/Sidebar/Sidebar'
import { useEffect, useState } from 'react'

export default function Home() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const DEFAULT_WIDTH = 320
  const MIN_WIDTH = 260
  const MAX_WIDTH = 480
  const [sidebarWidth, setSidebarWidth] = useState(DEFAULT_WIDTH)

  useEffect(() => {
    const storedWidth = typeof window !== 'undefined' ? window.localStorage.getItem('sidebar-width') : null
    if (storedWidth) {
      const parsed = parseInt(storedWidth, 10)
      if (!Number.isNaN(parsed)) {
        setSidebarWidth(Math.min(Math.max(parsed, MIN_WIDTH), MAX_WIDTH))
      }
    }
  }, [])

  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem('sidebar-width', String(sidebarWidth))
    }
  }, [sidebarWidth])

  const clampWidth = (next: number) => Math.min(Math.max(next, MIN_WIDTH), MAX_WIDTH)

  return (
    <div className="flex h-screen bg-secondary-50">
      {/* Sidebar */}
      <Sidebar
        open={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        width={sidebarWidth}
  onWidthChange={(value) => setSidebarWidth(clampWidth(value))}
        minWidth={MIN_WIDTH}
        maxWidth={MAX_WIDTH}
      />

      {/* Main Content */}
      <main className={`flex-1 flex flex-col transition-all duration-300 ${sidebarOpen ? 'ml-0' : 'ml-0'}`}>
        <ChatInterface />
      </main>
    </div>
  )
}
