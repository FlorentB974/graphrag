'use client'

import { useEffect, useState, useRef } from 'react'
import { api } from '@/lib/api'
import { DatabaseStats, ProcessingSummary } from '@/types'
import { TrashIcon } from '@heroicons/react/24/outline'
import { useChatStore } from '@/store/chatStore'

export default function DatabaseTab() {
  const [stats, setStats] = useState<DatabaseStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [processingState, setProcessingState] = useState<ProcessingSummary | null>(null)
  const [isStuck, setIsStuck] = useState(false)
  const wasProcessingRef = useRef(false)
  const lastUpdateTimestampRef = useRef<number>(Date.now())
  const selectDocument = useChatStore((state) => state.selectDocument)
  const clearSelectedDocument = useChatStore((state) => state.clearSelectedDocument)
  const selectedDocumentId = useChatStore((state) => state.selectedDocumentId)

  useEffect(() => {
    loadStats()

    // Listen for processing completion events to refresh stats automatically
    const handler = () => {
      loadStats()
    }

    if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
      window.addEventListener('documents:processed', handler)
      window.addEventListener('documents:processing-updated', handler)
      window.addEventListener('documents:uploaded', handler)
    }

    return () => {
      if (typeof window !== 'undefined' && typeof window.removeEventListener === 'function') {
        window.removeEventListener('documents:processed', handler)
        window.removeEventListener('documents:processing-updated', handler)
        window.removeEventListener('documents:uploaded', handler)
      }
    }
  }, [])

  // Poll for processing updates when active (without refreshing entire stats)
  useEffect(() => {
    const isProcessing = processingState?.is_processing || stats?.processing?.is_processing
    if (!isProcessing) return

    const interval = setInterval(async () => {
      try {
        const response = await api.getProcessingProgress()
        
        // Check if ALL processing finished (no pending documents and is_processing is false)
        const hasPendingDocs = response.global.pending_documents && response.global.pending_documents.length > 0
        const isStillProcessing = response.global.is_processing || hasPendingDocs
        
        if (!isStillProcessing) {
          console.log('Processing completed detected in polling - clearing state')
          // Clear processing state immediately
          setProcessingState(null)
          setIsStuck(false)
          wasProcessingRef.current = false
          // Force a final stats refresh
          await loadStats()
          return
        }
        
        setProcessingState(response.global)
        
        // Update timestamp on successful response with active progress
        lastUpdateTimestampRef.current = Date.now()
        setIsStuck(false)
        
        // Also fetch updated stats to get real-time counts
        const updatedStats = await api.getStats()
        
        // Update document progress in stats if we have them
        if (stats && response.global.pending_documents) {
          const updatedDocs = stats.documents.map(doc => {
            const progressMatch = response.global.pending_documents.find(
              p => p.document_id === doc.document_id
            )
            if (progressMatch) {
              return {
                ...doc,
                processing_status: progressMatch.status,
                processing_stage: progressMatch.stage || doc.processing_stage,
                processing_progress: progressMatch.progress_percentage,
                queue_position: progressMatch.queue_position
              }
            }
            return doc
          })
          
          // Merge updated documents with their progress AND update global stats
          setStats({ 
            ...updatedStats,
            documents: updatedDocs.map(doc => {
              // Find matching document in updated stats to get chunk count
              const freshDoc = updatedStats.documents.find((d: any) => d.document_id === doc.document_id)
              return freshDoc ? { ...doc, chunk_count: freshDoc.chunk_count } : doc
            }),
            processing: response.global 
          })
        } else {
          // Just update with fresh stats if no processing documents
          setStats({ ...updatedStats, processing: response.global })
        }
      } catch (error) {
        console.error('Failed to poll processing state:', error)
      }
    }, 1500) // Poll every 1.5s during processing

    return () => clearInterval(interval)
  }, [processingState?.is_processing, stats?.processing?.is_processing, stats])

  // Detect stuck/stale processing (server crash)
  useEffect(() => {
    const isProcessing = processingState?.is_processing || stats?.processing?.is_processing
    if (!isProcessing) return

    const checkStuck = setInterval(() => {
      const timeSinceUpdate = Date.now() - lastUpdateTimestampRef.current
      const STUCK_THRESHOLD = 30000 // 30 seconds
      
      if (timeSinceUpdate > STUCK_THRESHOLD) {
        console.warn('Processing appears stuck - no updates for 30s')
        setIsStuck(true)
      }
    }, 5000) // Check every 5 seconds

    return () => clearInterval(checkStuck)
  }, [processingState?.is_processing, stats?.processing?.is_processing])

  // Detect when processing completes and do a final refresh
  useEffect(() => {
    const isProcessing = processingState?.is_processing || stats?.processing?.is_processing || false
    
    // If we were processing before but not anymore, refresh everything and clear state
    if (wasProcessingRef.current && !isProcessing) {
      console.log('Processing completed, refreshing stats...')
      loadStats()
      // Clear processing state to hide progress indicators
      setTimeout(() => {
        setProcessingState(null)
        setIsStuck(false)
      }, 1000) // Small delay to ensure final stats are loaded
    }
    
    // Update the ref for next check
    wasProcessingRef.current = isProcessing
  }, [processingState?.is_processing, stats?.processing?.is_processing])

  const loadStats = async () => {
    try {
      setLoading(true)
      const data = await api.getStats()
      console.log('Loaded stats, is_processing:', data.processing?.is_processing, 'pending docs:', data.processing?.pending_documents?.length || 0)
      setStats(data)
      setProcessingState(data.processing || null)
      return data
    } catch (error) {
      console.error('Failed to load stats:', error)
      return null
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteDocument = async (documentId: string) => {
    if (!confirm('Delete this document and all its chunks?')) return

    try {
      await api.deleteDocument(documentId)
      const newStats = await loadStats()

      // If the deleted document was selected, switch selection to next available or fallback to chat
      if (selectedDocumentId === documentId) {
        if (newStats && newStats.documents && newStats.documents.length > 0) {
          // Find next document: try to find the document at the same index as the deleted one
          const idx = newStats.documents.findIndex((d: any) => d.document_id === documentId)
          // If not found (deleted), pick the next one at idx (same position) or the last one
          const pickIndex = Math.min(Math.max(0, idx), newStats.documents.length - 1)
          const nextDoc = newStats.documents[pickIndex]
          if (nextDoc) {
            selectDocument(nextDoc.document_id)
          } else {
            // No documents left
            clearSelectedDocument()
          }
        } else {
          // No documents left, go back to chat view
          clearSelectedDocument()
        }
      }
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
      const newStats = await loadStats()
      // After clearing the database, ensure the UI returns to chat view
      clearSelectedDocument()
    } catch (error) {
      console.error('Failed to clear database:', error)
    }
  }

  const handleClearStuckState = async () => {
    console.log('Clearing stuck state and refreshing...')
    setIsStuck(false)
    setProcessingState(null)
    lastUpdateTimestampRef.current = Date.now()
    wasProcessingRef.current = false
    await loadStats()
  }

  if (loading) {
    return <div className="text-center text-secondary-600">Loading...</div>
  }

  // Use processingState which gets updated via polling, or fallback to stats.processing
  const processingSummary = processingState || stats?.processing

  const formatStatus = (doc: any) => {
    const status = doc.processing_status
    const stage = doc.processing_stage
    const progress = typeof doc.processing_progress === 'number' ? Math.round(doc.processing_progress) : null

    if (status === 'processing') {
      return `Processing${stage ? ` – ${stage}` : ''}${progress !== null ? ` (${progress}%)` : ''}`
    }
    if (status === 'queued') {
      return 'Processing queued'
    }
    if (status === 'staged') {
      return 'Ready to process'
    }
    if (status === 'error') return 'Needs attention'
    return null
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

      {processingSummary?.is_processing && (
        <div className={`flex items-center gap-3 rounded-lg border px-4 py-3 text-sm shadow-sm ${
          isStuck 
            ? 'border-red-200 bg-red-50 text-red-700' 
            : 'border-secondary-200 bg-white text-secondary-700'
        }`}>
          <span className={`inline-flex h-2 w-2 rounded-full ${
            isStuck ? 'bg-red-500' : 'bg-primary-500 animate-pulse'
          }`} />
          <div className="flex-1">
            {isStuck ? (
              <>
                <p className="font-medium">Processing appears stuck</p>
                <p className="text-xs mt-1">No updates for 30+ seconds. Server may have crashed.</p>
              </>
            ) : (
              <p className="font-medium">Processing in progress…</p>
            )}
          </div>
          {isStuck && (
            <button
              onClick={handleClearStuckState}
              className="px-3 py-1 text-xs font-medium rounded bg-red-600 text-white hover:bg-red-700"
            >
              Clear
            </button>
          )}
        </div>
      )}

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
              const statusLabel = formatStatus(doc)
              const status = doc.processing_status
              const progress =
                typeof doc.processing_progress === 'number'
                  ? Math.max(0, Math.min(100, doc.processing_progress))
                  : null

              return (
                <div
                  key={index}
                  onClick={() => handleSelectDocument(doc.document_id)}
                  className={`card p-3 flex flex-col gap-2 transition-all cursor-pointer group ${
                    isActive
                      ? 'border-primary-300 shadow-primary-100 ring-1 ring-primary-100'
                      : 'hover:shadow-md'
                  }`}
                >
                  <div className="flex w-full items-start justify-between gap-3">
                    <div className="flex-1 min-w-0">
                      <div className="relative">
                        <p className="text-sm font-medium text-secondary-900 truncate">
                          {doc.filename}
                        </p>
                      </div>
                      <p className={`text-xs mt-1 ${isStuck && (status === 'queued' || status === 'staged') ? 'text-red-600' : 'text-secondary-600'}`}>
                        {status === 'queued' || status === 'staged' 
                          ? (isStuck ? 'Queue stuck - processing may have crashed' : 'Processing queued')
                          : `${doc.chunk_count} chunks`}
                      </p>
                      {statusLabel && status !== 'queued' && status !== 'staged' && (
                        <p
                          className={`text-[11px] mt-1 ${
                            status === 'error'
                              ? 'text-red-600'
                              : status === 'processing'
                              ? 'text-primary-600'
                              : 'text-secondary-500'
                          }`}
                        >
                          {statusLabel}
                        </p>
                      )}
                    </div>
                    <button
                      onClick={(event) => {
                        event.stopPropagation()
                        handleDeleteDocument(doc.document_id)
                      }}
                      className="text-red-600 hover:text-red-700 p-1 flex-shrink-0"
                      title={`Delete ${doc.filename}`}
                    >
                      <TrashIcon className="w-4 h-4" />
                    </button>
                  </div>

                  {status === 'processing' && progress !== null && (
                    <div className="w-full">
                      <div className="flex justify-between text-[10px] mb-1">
                        <span className={isStuck ? 'text-red-600' : 'text-secondary-500'}>
                          {isStuck ? 'Stuck - may need manual refresh' : (doc.processing_stage || 'Processing')}
                        </span>
                        <span className={isStuck ? 'text-red-600' : 'text-secondary-500'}>
                          {Math.round(progress)}%
                        </span>
                      </div>
                      <div className="h-1.5 w-full rounded-full bg-secondary-200">
                        <div
                          className={`h-1.5 rounded-full transition-all ${isStuck ? 'bg-red-500' : 'bg-primary-500'}`}
                          style={{ width: `${progress}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
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
