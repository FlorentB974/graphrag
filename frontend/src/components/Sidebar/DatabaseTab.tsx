'use client'

import { useEffect, useState, useRef } from 'react'
import { api } from '@/lib/api'
import { DatabaseStats, ProcessingSummary } from '@/types'
import { TrashIcon, DocumentArrowUpIcon, MagnifyingGlassIcon, XMarkIcon } from '@heroicons/react/24/outline'
import { useChatStore } from '@/store/chatStore'
import { showToast } from '@/components/Toast/ToastContainer'

export default function DatabaseTab() {
  const [stats, setStats] = useState<DatabaseStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [processingState, setProcessingState] = useState<ProcessingSummary | null>(null)
  const [isStuck, setIsStuck] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [uploadingFile, setUploadingFile] = useState(false)
  const [searchMode, setSearchMode] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [enableDeleteOps, setEnableDeleteOps] = useState(true)
  const searchInputRef = useRef<HTMLInputElement>(null)
  const wasProcessingRef = useRef(false)
  const lastUpdateTimestampRef = useRef<number>(Date.now())
  const selectDocument = useChatStore((state) => state.selectDocument)
  const clearSelectedDocument = useChatStore((state) => state.clearSelectedDocument)
  const selectedDocumentId = useChatStore((state) => state.selectedDocumentId)

  const handleFiles = async (files: FileList | File[]) => {
    setUploadingFile(true)

    try {
      const fileArray = Array.from(files)

      for (const file of fileArray) {
        await api.stageFile(file)
        showToast('success', `${file.name} uploaded`, 'Document queued for processing')
      }

      // Emit event to notify other components
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('documents:uploaded'))
      }
    } catch (error) {
      console.error('Failed to stage file:', error)
      showToast('error', 'Upload failed', error instanceof Error ? error.message : 'Failed to upload file')
    } finally {
      setUploadingFile(false)
    }
  }

  const handleFileInput = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    await handleFiles(files)
    e.target.value = ''
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const files = e.dataTransfer.files
    if (files && files.length > 0) {
      await handleFiles(files)
    }
  }

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

  // Focus search input when entering search mode
  useEffect(() => {
    if (searchMode && searchInputRef.current) {
      searchInputRef.current.focus()
    }
  }, [searchMode])

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
        
        // Also fetch updated stats to get real-time counts AND document_type updates
        const updatedStats = await api.getStats()
        
        // Update document progress in stats if we have them
        if (stats && response.global.pending_documents) {
          const updatedDocs = stats.documents.map(doc => {
            const progressMatch = response.global.pending_documents.find(
              p => p.document_id === doc.document_id
            )
            // Always get fresh data from updatedStats to ensure document_type and other fields are current
            const freshDoc = updatedStats.documents.find((d: any) => d.document_id === doc.document_id)
            if (progressMatch) {
              return {
                ...freshDoc,
                processing_status: progressMatch.status,
                processing_stage: progressMatch.stage || doc.processing_stage,
                processing_progress: progressMatch.progress_percentage,
                queue_position: progressMatch.queue_position
              }
            }
            return freshDoc || doc
          })
          
          // Merge updated documents with their progress AND update global stats
          setStats({ 
            ...updatedStats,
            documents: updatedDocs,
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
      setEnableDeleteOps(data.enable_delete_operations ?? true)
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

  const getFilteredDocuments = () => {
    if (!stats?.documents) return []
    if (!searchQuery.trim()) return stats.documents

    const query = searchQuery.toLowerCase()
    return stats.documents.filter(
      (doc) =>
        (doc.original_filename || doc.filename || '').toLowerCase().includes(query) ||
        ((doc as any).document_type || '').toLowerCase().includes(query)
    )
  }

  if (loading) {
    return <div className="text-center text-secondary-600 dark:text-secondary-400">Loading...</div>
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
    <div
      className={`relative transition-all ${isDragging ? 'ring-2 ring-primary-500 rounded-lg' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {isDragging && (
        <div className="absolute inset-0 bg-primary-50 border-2 border-dashed border-primary-500 rounded-lg flex items-center justify-center z-10">
          <div className="text-center">
            <DocumentArrowUpIcon className="w-8 h-8 text-primary-600 mx-auto mb-2" />
            <p className="text-sm font-medium text-primary-700">
              Drop files to upload
            </p>
          </div>
        </div>
      )}

      <div className="space-y-4">
        {/* Upload Button */}
        <div className="flex gap-2">
          <label className="flex-1 cursor-pointer">
            <div className={`button-primary py-2 px-3 text-center text-sm flex items-center justify-center gap-2 ${
              uploadingFile ? 'opacity-50 pointer-events-none' : ''
            }`}>
              <DocumentArrowUpIcon className="w-4 h-4" />
              <span>{uploadingFile ? 'Uploading...' : 'Upload Files'}</span>
            </div>
            <input
              type="file"
              className="hidden"
              onChange={handleFileInput}
              disabled={uploadingFile}
              accept=".pdf,.txt,.md,.doc,.docx,.ppt,.pptx,.xls,.xlsx"
              multiple
            />
          </label>
        </div>
      {/* Stats Cards */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-primary-50 dark:bg-primary-900/20 rounded-lg p-4">
          <div className="text-2xl font-bold text-primary-700 dark:text-primary-400">
            {stats?.total_documents || 0}
          </div>
          <div className="text-xs text-primary-600 dark:text-primary-400 mt-1">Documents</div>
        </div>
        <div className="bg-secondary-100 dark:bg-secondary-700 rounded-lg p-4">
          <div className="text-2xl font-bold text-secondary-700 dark:text-secondary-200">
            {stats?.total_chunks || 0}
          </div>
          <div className="text-xs text-secondary-600 dark:text-secondary-400 mt-1">Chunks</div>
        </div>
        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
          <div className="text-2xl font-bold text-green-700 dark:text-green-400">
            {stats?.total_entities || 0}
          </div>
          <div className="text-xs text-green-600 dark:text-green-400 mt-1">Entities</div>
        </div>
        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4">
          <div className="text-2xl font-bold text-purple-700 dark:text-purple-400">
            {stats?.total_relationships || 0}
          </div>
          <div className="text-xs text-purple-600 dark:text-purple-400 mt-1">Relationships</div>
        </div>
      </div>

      {processingSummary?.is_processing && (
        <div className={`flex items-center gap-3 rounded-lg border px-4 py-3 text-sm shadow-sm ${
          isStuck 
            ? 'border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-900/20 dark:text-red-400' 
            : 'border-secondary-200 bg-white dark:bg-secondary-800 text-secondary-700 dark:text-secondary-300'
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
          {!searchMode ? (
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-medium text-secondary-900 dark:text-secondary-50">Documents</h3>
                <button
                  onClick={() => setSearchMode(true)}
                  className="text-secondary-600 dark:text-secondary-400 hover:text-secondary-900 dark:hover:text-secondary-200 p-1"
                  title="Search documents"
                >
                  <MagnifyingGlassIcon className="w-4 h-4" />
                </button>
              </div>
              {enableDeleteOps && (
                <button
                  onClick={handleClearDatabase}
                  className="text-xs text-red-600 hover:text-red-700"
                >
                  Clear All
                </button>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <div className="flex-1 relative">
                <input
                  ref={searchInputRef}
                  type="text"
                  placeholder="Search documents..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Escape') {
                      setSearchMode(false)
                      setSearchQuery('')
                    }
                  }}
                  className="w-full px-3 py-2 text-sm border border-secondary-300 rounded-md focus:outline-none focus:ring-1 focus:ring-primary-500"
                />
              </div>
              <button
                onClick={() => {
                  setSearchMode(false)
                  setSearchQuery('')
                }}
                className="text-secondary-600 hover:text-secondary-900 dark:text-secondary-400 dark:hover:text-secondary-200 p-1"
                title="Close search"
              >
                <XMarkIcon className="w-4 h-4" />
              </button>
            </div>
          )}


          <div className="space-y-2">
            {getFilteredDocuments().map((doc, index) => {
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
                  draggable
                  onDragStart={(e) => {
                    e.dataTransfer.effectAllowed = 'copy'
                    e.dataTransfer.setData('application/json', JSON.stringify({
                      type: 'document',
                      document_id: doc.document_id,
                      filename: doc.original_filename || doc.filename,
                    }))
                  }}
                  onClick={() => handleSelectDocument(doc.document_id)}
                  className={`card p-3 flex flex-col gap-2 transition-all cursor-move group ${
                    isActive
                      ? 'border-primary-300 shadow-primary-100 ring-1 ring-primary-100'
                      : 'hover:shadow-md hover:cursor-grab active:cursor-grabbing'
                  }`}
                >
                  <div className="flex w-full items-start justify-between gap-3">
                    <div className="flex-1 min-w-0">
                      <div className="relative">
                        <p className="text-sm font-medium text-secondary-900 dark:text-secondary-50 truncate">
                          {doc.original_filename || doc.filename}
                        </p>
                      </div>
                      <p className={`text-xs mt-1 ${isStuck && (status === 'queued' || status === 'staged') ? 'text-red-600 dark:text-red-400' : 'text-secondary-600 dark:text-secondary-400'}`}>
                        {status === 'queued' || status === 'staged' 
                          ? (isStuck ? 'Queue stuck - processing may have crashed' : 'Processing queued')
                          : (doc as any).document_type 
                          ? (doc as any).document_type.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase())
                          : 'Unknown type'}
                      </p>
                      {statusLabel && status !== 'queued' && status !== 'staged' && (
                        <p
                          className={`text-[11px] mt-1 ${
                            status === 'error'
                              ? 'text-red-600'
                              : status === 'processing'
                              ? 'text-primary-600'
                              : 'text-secondary-500 dark:text-secondary-400'
                          }`}
                        >
                          {statusLabel}
                        </p>
                      )}
                    </div>
                    {enableDeleteOps && (
                      <button
                        onClick={(event) => {
                          event.stopPropagation()
                          handleDeleteDocument(doc.document_id)
                        }}
                        className="text-red-600 hover:text-red-700 p-1 flex-shrink-0"
                        title={`Delete ${doc.original_filename || doc.filename}`}
                      >
                        <TrashIcon className="w-4 h-4" />
                      </button>
                    )}
                  </div>

                  {status === 'processing' && progress !== null && (
                    <div className="w-full">
                      <div className="flex justify-between text-[10px] mb-1">
                        <span className={isStuck ? 'text-red-600' : 'text-secondary-500 dark:text-secondary-400'}>
                          {isStuck ? 'Stuck - may need manual refresh' : (doc.processing_stage || 'Processing')}
                        </span>
                        <span className={isStuck ? 'text-red-600' : 'text-secondary-500 dark:text-secondary-400'}>
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
            {searchQuery.trim() && getFilteredDocuments().length === 0 && (
              <div className="text-center text-secondary-600 dark:text-secondary-400 py-6">
                <p className="text-sm">No documents match &quot;{searchQuery}&quot;</p>
              </div>
            )}
          </div>
        </>
      )}

      {stats && stats.documents.length === 0 && (
        <div className="text-center text-secondary-600 dark:text-secondary-400 py-8">
          <p>No documents in database</p>
          <p className="text-xs mt-1">Upload documents to get started</p>
        </div>
      )}
      </div>
    </div>
  )
}
