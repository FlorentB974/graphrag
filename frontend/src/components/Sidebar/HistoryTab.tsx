'use client'

import { useEffect, useState } from 'react'
import { api } from '@/lib/api'
import { ChatSession } from '@/types'
import { formatDistanceToNow } from 'date-fns'
import { TrashIcon } from '@heroicons/react/24/outline'
import { useChatStore } from '@/store/chatStore'

export default function HistoryTab() {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [loading, setLoading] = useState(true)
  const loadSession = useChatStore((state) => state.loadSession)
  const setSessionId = useChatStore((state) => state.setSessionId)
  const clearChat = useChatStore((state) => state.clearChat)
  const historyRefreshKey = useChatStore((state) => state.historyRefreshKey)
  const activeSessionId = useChatStore((state) => state.sessionId)

  useEffect(() => {
    loadSessions()
  }, [])

  // Reload history when another part of the app signals a refresh (e.g., New Chat)
  useEffect(() => {
    // ignore initial mount because loadSessions already ran
    if (historyRefreshKey > 0) {
      loadSessions()
    }
  }, [historyRefreshKey])

  const loadSessions = async () => {
    try {
      setLoading(true)
      const data = await api.getHistory()
      setSessions(data)
    } catch (error) {
      console.error('Failed to load history:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (sessionId: string) => {
    if (!confirm('Delete this conversation?')) return

    try {
      await api.deleteConversation(sessionId)
      await loadSessions()
    } catch (error) {
      console.error('Failed to delete conversation:', error)
    }
  }

  const handleClearAll = async () => {
    if (!confirm('Clear all conversation history?')) return

    try {
      await api.clearHistory()
      await loadSessions()
    } catch (error) {
      console.error('Failed to clear history:', error)
    }
  }

  if (loading) {
    return <div className="text-center text-secondary-600">Loading...</div>
  }

  return (
    <div className="space-y-4">
      {sessions.length > 0 && (
        <button onClick={handleClearAll} className="button-secondary w-full text-sm">
          Clear All History
        </button>
      )}

      {sessions.length === 0 ? (
        <div className="text-center text-secondary-600 py-8">
          <p>No conversation history yet</p>
        </div>
      ) : (
        <div className="space-y-3">
          {sessions.map((session) => (
            <div
              key={session.session_id}
              className={`card p-4 hover:shadow-md transition-shadow cursor-pointer ${
                activeSessionId === session.session_id
                  ? 'border-primary-400 shadow-primary-100'
                  : ''
              }`}
              onClick={async () => {
                if (activeSessionId === session.session_id) {
                  return
                }
                await loadSession(session.session_id)
                setSessionId(session.session_id)
              }}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <p className="text-sm text-secondary-900 line-clamp-2">
                    {session.preview || 'New conversation'}
                  </p>
                  <p className="text-xs text-secondary-600 mt-1">
                    {session.message_count} messages •{' '}
                    {formatDistanceToNow(new Date(session.updated_at), { addSuffix: true })}
                  </p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    handleDelete(session.session_id)
                    if (activeSessionId === session.session_id) {
                      clearChat()
                    }
                  }}
                  className="text-red-600 hover:text-red-700 p-1"
                >
                  <TrashIcon className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
