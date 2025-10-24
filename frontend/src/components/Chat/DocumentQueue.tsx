'use client'

import { useState, useEffect, useCallback } from 'react'
import { api } from '@/lib/api'
import {
  PlayIcon,
  TrashIcon,
  CheckCircleIcon,
  XCircleIcon,
  DocumentIcon,
} from '@heroicons/react/24/outline'
import type { StagedDocument, ProcessProgress } from '@/types/upload'

interface DocumentQueueProps {
  refreshKey?: number
  onRefresh?: () => void
}

export default function DocumentQueue({ refreshKey, onRefresh }: DocumentQueueProps) {
  const [stagedDocuments, setStagedDocuments] = useState<StagedDocument[]>([])
  const [processingProgress, setProcessingProgress] = useState<
    Record<string, ProcessProgress>
  >({})
  const [isProcessing, setIsProcessing] = useState(false)
  const [shouldHideQueue, setShouldHideQueue] = useState(false)

  // Load staged documents
  const loadStagedDocuments = useCallback(async () => {
    try {
      const response = await api.getStagedDocuments()
      setStagedDocuments(response.documents || [])
    } catch (error) {
      console.error('Failed to load staged documents:', error)
    }
  }, [])

  // Load on mount and when refreshKey changes
  useEffect(() => {
    void loadStagedDocuments()
  }, [loadStagedDocuments, refreshKey])

  // Poll for processing progress
  useEffect(() => {
    if (!isProcessing) return

    const interval = setInterval(async () => {
      try {
        const response = await api.getProcessingProgress()
        const progressList: ProcessProgress[] = response.progress || []
        const globalState = response.global

        const progressMap: Record<string, ProcessProgress> = {}
        progressList.forEach((p) => {
          progressMap[p.file_id] = p
        })

        setProcessingProgress(progressMap)
        setIsProcessing(Boolean(globalState?.is_processing))

        // Check if all processing is complete
        const allComplete = progressList.every(
          (p) => p.status === 'completed' || p.status === 'error'
        )

        if (allComplete && progressList.length > 0) {
          setIsProcessing(false)
          setShouldHideQueue(true)
          // Reload staged documents to remove completed ones
          setTimeout(() => {
            void loadStagedDocuments()
            setProcessingProgress({})
            if (onRefresh) {
              onRefresh()
            }

            // Notify other parts of the UI (e.g. Database tab) that documents were processed
            try {
              if (typeof window !== 'undefined' && typeof window.dispatchEvent === 'function') {
                window.dispatchEvent(new CustomEvent('documents:processed'))
              }
            } catch (e) {
              // ignore
            }
          }, 2000)
        }
      } catch (error) {
        console.error('Failed to fetch progress:', error)
      }
    }, 500)

    return () => clearInterval(interval)
  }, [isProcessing, loadStagedDocuments, onRefresh])

  const handleDeleteDocument = async (fileId: string) => {
    try {
      await api.deleteStagedDocument(fileId)
      await loadStagedDocuments()
    } catch (error) {
      console.error('Failed to delete document:', error)
    }
  }

  const handleProcessAll = async () => {
    if (stagedDocuments.length === 0) return

    try {
      setIsProcessing(true)
      setShouldHideQueue(true)
      const fileIds = stagedDocuments.map((doc) => doc.file_id)
      await api.processDocuments(fileIds)
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('documents:processing-updated'))
      }
    } catch (error) {
      console.error('Failed to start processing:', error)
      setIsProcessing(false)
      setShouldHideQueue(false)
    }
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  useEffect(() => {
    if (stagedDocuments.length > 0 && !isProcessing) {
      setShouldHideQueue(false)
    }
  }, [stagedDocuments.length, isProcessing])

  const hasDocuments = stagedDocuments.length > 0
  const shouldRenderQueue = hasDocuments && !shouldHideQueue

  // Don't render if no documents or queue is hidden
  if (!shouldRenderQueue) {
    return null
  }

  return (
    <div className="border-t border-secondary-200 bg-secondary-50 px-6 py-4">
      <div className="max-w-4xl mx-auto">
        {/* Header with Process Button */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <DocumentIcon className="w-5 h-5 text-secondary-600" />
            <h3 className="text-sm font-semibold text-secondary-900">
              Documents Queue ({stagedDocuments.length})
            </h3>
          </div>
          <button
            onClick={handleProcessAll}
            disabled={isProcessing}
            className="button-primary text-sm px-4 py-2 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <PlayIcon className="w-4 h-4" />
            Process All
          </button>
        </div>

        {/* Document List */}
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {stagedDocuments.map((doc) => {
            const progress = processingProgress[doc.file_id]
            const isDocProcessing = progress?.status === 'processing'
            const isQueued = progress?.status === 'queued'
            const isCompleted = progress?.status === 'completed'
            const hasError = progress?.status === 'error'

            return (
              <div
                key={doc.file_id}
                className="bg-white border border-secondary-200 rounded-lg p-3 shadow-sm"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1 min-w-0 mr-3">
                    <p className="text-sm font-medium text-secondary-900 truncate">
                      {doc.filename}
                    </p>
                    <p className="text-xs text-secondary-500">
                      {formatFileSize(doc.file_size)}
                    </p>
                  </div>
                  {!isProcessing && (
                    <button
                      onClick={() => handleDeleteDocument(doc.file_id)}
                      className="p-1 text-secondary-400 hover:text-red-600 transition-colors flex-shrink-0"
                      title="Remove document"
                    >
                      <TrashIcon className="w-4 h-4" />
                    </button>
                  )}
                </div>

                {/* Queue State */}
                {isQueued && (
                  <div className="text-xs text-secondary-600">
                    Processing queued
                  </div>
                )}

                {/* Progress Bar */}
                {isDocProcessing && progress && (
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs text-secondary-600">
                      <span>
                        {progress.chunks_processed} / {progress.total_chunks}{' '}
                        chunks
                      </span>
                      <span>{progress.progress_percentage.toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-secondary-200 rounded-full h-2">
                      <div
                        className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                        style={{
                          width: `${progress.progress_percentage}%`,
                        }}
                      />
                    </div>
                  </div>
                )}

                {/* Completed State */}
                {isCompleted && (
                  <div className="flex items-center text-xs text-green-600">
                    <CheckCircleIcon className="w-4 h-4 mr-1" />
                    Processing completed
                  </div>
                )}

                {/* Error State */}
                {hasError && progress?.error && (
                  <div className="flex items-start text-xs text-red-600">
                    <XCircleIcon className="w-4 h-4 mr-1 flex-shrink-0" />
                    <span>{progress.error}</span>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
