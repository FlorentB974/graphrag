'use client'

import { useEffect, useState } from 'react'
import { api } from '@/lib/api'
import { DatabaseStats } from '@/types'
import { TrashIcon } from '@heroicons/react/24/outline'
import { useChatStore } from '@/store/chatStore'

export default function DatabaseTab() {
  const [stats, setStats] = useState<DatabaseStats | null>(null)
  const [loading, setLoading] = useState(true)
  const selectDocument = useChatStore((state) => state.selectDocument)
  const selectedDocumentId = useChatStore((state) => state.selectedDocumentId)

  useEffect(() => {
    loadStats()

    // Listen for processing completion events to refresh stats automatically
    const handler = () => {
      loadStats()
    }

    if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
      window.addEventListener('documents:processed', handler)
    }

    return () => {
      if (typeof window !== 'undefined' && typeof window.removeEventListener === 'function') {
        window.removeEventListener('documents:processed', handler)
      }
    }
  }, [])

  const loadStats = async () => {
    try {
      setLoading(true)
      const data = await api.getStats()
      setStats(data)
    } catch (error) {
      console.error('Failed to load stats:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteDocument = async (documentId: string) => {
    if (!confirm('Delete this document and all its chunks?')) return

    try {
      await api.deleteDocument(documentId)
      await loadStats()
    } catch (error) {
      console.error('Failed to delete document:', error)
    }
  }

  const handleSelectDocument = (documentId: string) => {
    selectDocument(documentId)
  }

  const handleClearDatabase = async () => {
    if (!confirm('Clear the entire database? This cannot be undone.')) return

    try {
      await api.clearDatabase()
      await loadStats()
    } catch (error) {
      console.error('Failed to clear database:', error)
    }
  }

  if (loading) {
    return <div className="text-center text-secondary-600">Loading...</div>
  }

  return (
    <div className="space-y-4">
      {/* Stats Cards */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-primary-50 rounded-lg p-4">
          <div className="text-2xl font-bold text-primary-700">
            {stats?.total_documents || 0}
          </div>
          <div className="text-xs text-primary-600 mt-1">Documents</div>
        </div>
        <div className="bg-secondary-100 rounded-lg p-4">
          <div className="text-2xl font-bold text-secondary-700">
            {stats?.total_chunks || 0}
          </div>
          <div className="text-xs text-secondary-600 mt-1">Chunks</div>
        </div>
        <div className="bg-green-50 rounded-lg p-4">
          <div className="text-2xl font-bold text-green-700">
            {stats?.total_entities || 0}
          </div>
          <div className="text-xs text-green-600 mt-1">Entities</div>
        </div>
        <div className="bg-purple-50 rounded-lg p-4">
          <div className="text-2xl font-bold text-purple-700">
            {stats?.total_relationships || 0}
          </div>
          <div className="text-xs text-purple-600 mt-1">Relationships</div>
        </div>
      </div>

      {/* Documents List */}
      {stats && stats.documents.length > 0 && (
        <>
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-secondary-900">Documents</h3>
            <button
              onClick={handleClearDatabase}
              className="text-xs text-red-600 hover:text-red-700"
            >
              Clear All
            </button>
          </div>

          <div className="space-y-2">
            {stats.documents.map((doc, index) => {
              const isActive = doc.document_id === selectedDocumentId
              return (
              <div
                key={index}
                onClick={() => handleSelectDocument(doc.document_id)}
                className={`card p-3 flex items-center justify-between cursor-pointer transition-colors ${
                  isActive ? 'ring-2 ring-primary-500 border-primary-500' : 'hover:border-primary-200'
                }`}
              >
                <div className="flex-1 min-w-0 pr-8">
                    <div className="relative">
                      <p className="text-sm font-medium text-secondary-900 truncate">
                        {doc.filename}
                      </p>
                      {/* fading gradient to the right so long names visually fade out */}
                    </div>
                    <p className="text-xs text-secondary-600 mt-1">
                      {doc.chunk_count} chunks
                    </p>
                  </div>
                  <button
                    onClick={(event) => {
                      event.stopPropagation()
                      handleDeleteDocument(doc.document_id)
                    }}
                    className="text-red-600 hover:text-red-700 p-1 ml-2 flex-shrink-0"
                    title={`Delete ${doc.filename}`}
                  >
                    <TrashIcon className="w-4 h-4" />
                  </button>
              </div>
            )})}
          </div>
        </>
      )}

      {stats && stats.documents.length === 0 && (
        <div className="text-center text-secondary-600 py-8">
          <p>No documents in database</p>
          <p className="text-xs mt-1">Upload documents to get started</p>
        </div>
      )}
    </div>
  )
}
