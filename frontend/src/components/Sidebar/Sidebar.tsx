'use client'

import { useCallback, useState, useEffect } from 'react'
import type { MouseEvent as ReactMouseEvent, TouchEvent as ReactTouchEvent } from 'react'
import {
  Bars3Icon,
  XMarkIcon,
  ClockIcon,
  CircleStackIcon,
} from '@heroicons/react/24/outline'
import { ChevronLeftIcon, ChevronRightIcon } from '@heroicons/react/24/outline'
import HistoryTab from './HistoryTab'
import DatabaseTab from './DatabaseTab'
import { useChatStore } from '@/store/chatStore'

interface SidebarProps {
  open: boolean
  onToggle: () => void
  width: number
  onWidthChange: (width: number) => void
  minWidth?: number
  maxWidth?: number
}

export default function Sidebar({
  open,
  onToggle,
  width,
  onWidthChange,
  minWidth = 260,
  maxWidth = 480,
}: SidebarProps) {
  const [activeTab, setActiveTab] = useState<'history' | 'database'>('history')
  const [isResizing, setIsResizing] = useState(false)
  const [collapsed, setCollapsed] = useState(false)
  const isConnected = useChatStore((state) => state.isConnected)

  const tabs = [
    { id: 'history' as const, label: 'History', icon: ClockIcon },
    { id: 'database' as const, label: 'Database', icon: CircleStackIcon },
  ]

  const resizeWithinBounds = useCallback(
    (nextWidth: number) => {
      const clamped = Math.min(Math.max(nextWidth, minWidth), maxWidth)
      onWidthChange(clamped)
    },
    [maxWidth, minWidth, onWidthChange]
  )

  const handleResizeStart = useCallback(
    (startPosition: number) => {
      const startWidth = width

      const handlePointerMove = (eventPosition: number) => {
        const delta = eventPosition - startPosition
        resizeWithinBounds(startWidth + delta)
      }

      const handleMouseMove = (event: MouseEvent) => {
        handlePointerMove(event.clientX)
      }

      const handleTouchMove = (event: TouchEvent) => {
        if (event.touches.length > 0) {
          event.preventDefault()
          handlePointerMove(event.touches[0].clientX)
        }
      }

      const stopResizing = () => {
        setIsResizing(false)
        document.body.style.removeProperty('cursor')
        document.body.style.removeProperty('user-select')
        window.removeEventListener('mousemove', handleMouseMove)
        window.removeEventListener('touchmove', handleTouchMove)
        window.removeEventListener('mouseup', stopResizing)
        window.removeEventListener('touchend', stopResizing)
        window.removeEventListener('touchcancel', stopResizing)
      }

      setIsResizing(true)
      document.body.style.cursor = 'col-resize'
      document.body.style.userSelect = 'none'
      window.addEventListener('mousemove', handleMouseMove)
      window.addEventListener('touchmove', handleTouchMove, { passive: false })
      window.addEventListener('mouseup', stopResizing)
      window.addEventListener('touchend', stopResizing)
      window.addEventListener('touchcancel', stopResizing)
    },
    [resizeWithinBounds, width]
  )

  const onMouseDown = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      event.preventDefault()
      handleResizeStart(event.clientX)
    },
    [handleResizeStart]
  )

  const onTouchStart = useCallback(
    (event: ReactTouchEvent<HTMLDivElement>) => {
      event.preventDefault()
      if (event.touches.length > 0) {
        handleResizeStart(event.touches[0].clientX)
      }
    },
    [handleResizeStart]
  )

  return (
    <>
      {/* Mobile toggle button */}
      <button
        onClick={onToggle}
        className="fixed top-4 left-4 z-50 lg:hidden button-primary p-2"
      >
        {open ? <XMarkIcon className="w-6 h-6" /> : <Bars3Icon className="w-6 h-6" />}
      </button>

      {/* Sidebar */}
      <aside
        className={`fixed lg:static inset-y-0 left-0 z-40 w-80 bg-white dark:bg-secondary-800 border-r border-secondary-200 dark:border-secondary-700 transform transition-all duration-300 ease-in-out ${
          open ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        } ${isResizing ? 'no-transition' : ''} relative`}
        style={{ width: `${collapsed ? 72 : width}px` }}
      >
        <div className="flex flex-col h-full">
          {/* Collapse button (desktop) */}
          <button
            onClick={() => setCollapsed((c) => !c)}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            className="hidden lg:flex absolute top-4 right-4 z-50 items-center justify-center p-2 rounded bg-white dark:bg-secondary-700 border border-secondary-200 dark:border-secondary-600 hover:bg-secondary-50 dark:hover:bg-secondary-600"
          >
            {collapsed ? (
              <ChevronRightIcon className="w-5 h-5" />
            ) : (
              <ChevronLeftIcon className="w-5 h-5" />
            )}
          </button>

          {/* When collapsed we hide the rest of the content entirely */}
          {!collapsed && (
            <>
              {/* Logo/Brand */}
              <div className="p-6 border-b border-secondary-200 dark:border-secondary-700">
                <h1 className="text-xl font-bold text-secondary-900 dark:text-secondary-50">🚀 GraphRAG</h1>
                <p className="text-sm text-secondary-600 dark:text-secondary-400 mt-1">Document Intelligence</p>
              </div>

              {/* Tabs */}
              <div className={`flex border-b border-secondary-200 dark:border-secondary-700 transition-all duration-300 ${
                !isConnected ? 'blur-sm pointer-events-none' : ''
              }`}>
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex-1 flex items-center justify-center py-3 text-sm font-medium transition-colors ${
                      activeTab === tab.id
                        ? 'text-primary-600 dark:text-primary-400 border-b-2 border-primary-600 dark:border-primary-400'
                        : 'text-secondary-600 dark:text-secondary-400 hover:text-secondary-900 dark:hover:text-secondary-200'
                    }`}
                  >
                    <tab.icon className="w-5 h-5 mr-1" />
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Tab Content */}
              <div className={`flex-1 overflow-y-auto p-6 transition-all duration-300 ${
                !isConnected ? 'blur-sm pointer-events-none' : ''
              }`}>
                <div key={activeTab} className="tab-content">
                  {activeTab === 'history' && <HistoryTab />}
                  {activeTab === 'database' && <DatabaseTab />}
                </div>
              </div>
            </>
          )}
        </div>

        {/* Resize handle (hidden when collapsed) */}
        {!collapsed && (
          <div
            className={`hidden lg:flex absolute top-0 -right-2 h-full w-4 items-center justify-center cursor-col-resize ${
              isResizing ? 'bg-primary-100 dark:bg-primary-900/60' : 'bg-transparent'
            }`}
            onMouseDown={onMouseDown}
            onTouchStart={onTouchStart}
          >
            <div className="h-16 w-1 rounded bg-secondary-300 dark:bg-secondary-600" />
          </div>
        )}
      </aside>

      {/* Overlay for mobile */}
      {open && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-30 lg:hidden"
          onClick={onToggle}
        />
      )}
    </>
  )
}
