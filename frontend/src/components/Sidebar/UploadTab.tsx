'use client'

import { useState, useEffect, useCallback } from 'react'
import { api } from '@/lib/api'
import {
  CloudArrowUpIcon,
  CheckCircleIcon,
  XCircleIcon,
  PlayIcon,
  TrashIcon,
} from '@heroicons/react/24/outline'
import type { StagedDocument, ProcessProgress } from '@/types/upload'

export default function UploadTab() {
  const [stagedDocuments, setStagedDocuments] = useState<StagedDocument[]>([])
  const [processingProgress, setProcessingProgress] = useState<
    Record<string, ProcessProgress>
  >({})
  const [isDragging, setIsDragging] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)

  // Load staged documents on mount
  useEffect(() => {
    loadStagedDocuments()
  }, [])

  // Poll for processing progress
  useEffect(() => {
    if (!isProcessing) return

    const interval = setInterval(async () => {
      try {
        const response = await api.getProcessingProgress()
        const progressList: ProcessProgress[] = response.progress || []

        const progressMap: Record<string, ProcessProgress> = {}
        progressList.forEach((p) => {
          progressMap[p.file_id] = p
        })

        setProcessingProgress(progressMap)

        // Check if all processing is complete
        const allComplete = progressList.every(
          (p) => p.status === 'completed' || p.status === 'error'
        )

        if (allComplete && progressList.length > 0) {
          setIsProcessing(false)
          // Reload staged documents to remove completed ones
          setTimeout(() => {
            loadStagedDocuments()
            setProcessingProgress({})
          }, 2000)
        }
      } catch (error) {
        console.error('Failed to fetch progress:', error)
      }
    }, 500)

    return () => clearInterval(interval)
  }, [isProcessing])

  const loadStagedDocuments = async () => {
    try {
      const response = await api.getStagedDocuments()
      setStagedDocuments(response.documents || [])
    } catch (error) {
      console.error('Failed to load staged documents:', error)
    }
  }

  const handleFiles = async (files: FileList | File[]) => {
    setUploadError(null)

    const fileArray = Array.from(files)

    for (const file of fileArray) {
      try {
        const result = await api.stageFile(file)

        if (result.status === 'staged') {
          // Reload staged documents
          await loadStagedDocuments()
        } else {
          setUploadError(result.error || 'Failed to stage file')
        }
      } catch (error) {
        setUploadError(
          error instanceof Error ? error.message : 'Failed to stage file'
        )
      }
    }
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    await handleFiles(files)
    e.target.value = ''
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const files = e.dataTransfer.files
    if (files && files.length > 0) {
      await handleFiles(files)
    }
  }

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
      const fileIds = stagedDocuments.map((doc) => doc.file_id)
      await api.processDocuments(fileIds)
    } catch (error) {
      console.error('Failed to start processing:', error)
      setIsProcessing(false)
      setUploadError(
        error instanceof Error ? error.message : 'Failed to start processing'
      )
    }
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  return (
    <div className="space-y-4">
      {/* Drop Zone */}
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          isDragging
            ? 'border-primary-500 bg-primary-50'
            : 'border-secondary-300 hover:border-primary-500'
        } ${isProcessing ? 'opacity-50 pointer-events-none' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          type="file"
          id="file-upload"
          className="hidden"
          onChange={handleFileUpload}
          disabled={isProcessing}
          accept=".pdf,.txt,.md,.doc,.docx,.ppt,.pptx,.xls,.xlsx"
          multiple
        />
        <label
          htmlFor="file-upload"
          className="cursor-pointer flex flex-col items-center"
        >
          <CloudArrowUpIcon className="w-12 h-12 text-secondary-400 mb-3" />
          <span className="text-sm font-medium text-secondary-700">
            {isDragging
              ? 'Drop files here'
              : 'Click to upload or drag and drop'}
          </span>
          <span className="text-xs text-secondary-500 mt-1">
            PDF, DOCX, TXT, MD, PPT, XLS
          </span>
        </label>
      </div>

      {/* Error Message */}
      {uploadError && (
        <div className="p-4 rounded-lg flex items-start bg-red-50 text-red-800">
          <XCircleIcon className="w-5 h-5 mr-2 flex-shrink-0" />
          <p className="text-sm">{uploadError}</p>
        </div>
      )}

      {/* Staged Documents Queue */}
      {stagedDocuments.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-secondary-700">
              Documents Queue ({stagedDocuments.length})
            </h3>
            <button
              onClick={handleProcessAll}
              disabled={isProcessing}
              className="button-primary text-xs px-3 py-1.5 flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <PlayIcon className="w-4 h-4" />
              Process All
            </button>
          </div>

          <div className="space-y-2">
            {stagedDocuments.map((doc) => {
              const progress = processingProgress[doc.file_id]
              const isDocProcessing = progress?.status === 'processing'
              const isCompleted = progress?.status === 'completed'
              const hasError = progress?.status === 'error'

              return (
                <div
                  key={doc.file_id}
                  className="border border-secondary-200 rounded-lg p-3 bg-white"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1 min-w-0">
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
                        className="ml-2 p-1 text-secondary-400 hover:text-red-600 transition-colors"
                      >
                        <TrashIcon className="w-4 h-4" />
                      </button>
                    )}
                  </div>

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
                      Completed
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
      )}

      {/* Info Section */}
      <div className="text-xs text-secondary-600 space-y-1">
        <p className="font-medium">Supported formats:</p>
        <ul className="list-disc list-inside space-y-0.5 ml-2">
          <li>PDF documents</li>
          <li>Word documents (.doc, .docx)</li>
          <li>PowerPoint (.ppt, .pptx)</li>
          <li>Excel (.xls, .xlsx)</li>
          <li>Text files (.txt, .md)</li>
        </ul>
      </div>
    </div>
  )
}

