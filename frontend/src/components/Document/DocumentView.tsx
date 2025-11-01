"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ArrowLeftIcon,
  DocumentTextIcon,
  Squares2X2Icon,
  MagnifyingGlassIcon,
  ClipboardIcon,
  ExclamationCircleIcon,
} from '@heroicons/react/24/outline'
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/solid'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'
import { motion, AnimatePresence } from 'framer-motion'
import { api } from '@/lib/api'
import type {
  DocumentChunk,
  DocumentDetails,
  DocumentEntity,
  RelatedDocument,
} from '@/types'
import type { ProcessingGlobalSummary } from '@/types/upload'
import { useChatStore } from '@/store/chatStore'
import DocumentPreview from './DocumentPreview'

interface PreviewState {
  url: string | null
  mimeType?: string
  isLoading: boolean
  error?: string | null
  objectUrl?: string | null
}

const initialPreviewState: PreviewState = {
  url: null,
  isLoading: false,
  mimeType: undefined,
  error: null,
  objectUrl: null,
}

export default function DocumentView() {
  const selectedDocumentId = useChatStore((state) => state.selectedDocumentId)
  const selectedChunkId = useChatStore((state) => state.selectedChunkId)
  const clearSelectedDocument = useChatStore((state) => state.clearSelectedDocument)
  const clearSelectedChunk = useChatStore((state) => state.clearSelectedChunk)
  const selectDocument = useChatStore((state) => state.selectDocument)

  const [documentData, setDocumentData] = useState<DocumentDetails | null>(null)
  const [hasPreview, setHasPreview] = useState<boolean | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expandedChunks, setExpandedChunks] = useState<Record<string | number, boolean>>({})
  const [previewState, setPreviewState] = useState<PreviewState>(initialPreviewState)
  const [showAllChunks, setShowAllChunks] = useState(false)
  const [showAllEntities, setShowAllEntities] = useState<Record<string, boolean>>({})
  const [processingState, setProcessingState] = useState<ProcessingGlobalSummary | null>(null)
  const [actionMessage, setActionMessage] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [isActionPending, setIsActionPending] = useState(false)
  const prevProcessingRef = useRef(false)
  const [isEditingHashtags, setIsEditingHashtags] = useState(false)
  const [newHashtagInput, setNewHashtagInput] = useState('')

  const refreshProcessingState = useCallback(async () => {
    try {
      const response = await api.getProcessingProgress()
      setProcessingState(response.global)
    } catch (stateError) {
      console.error('Failed to load processing state', stateError)
    }
  }, [])

  const CHUNKS_LIMIT = 10
  const ENTITIES_PER_TYPE_LIMIT = 5

  useEffect(() => {
    let isSubscribed = true

    const fetchDocument = async (documentId: string) => {
      setIsLoading(true)
      setError(null)
      setDocumentData(null)
      setPreviewState(initialPreviewState)
      setShowAllChunks(false)
      setShowAllEntities({})

      try {
        const data = await api.getDocument(documentId)
        if (isSubscribed) {
          setDocumentData(data)
          setHasPreview(null)
        }
      } catch (fetchError) {
        if (isSubscribed) {
          setError(fetchError instanceof Error ? fetchError.message : 'Failed to load document')
        }
      } finally {
        if (isSubscribed) {
          setIsLoading(false)
        }
      }
    }

    if (selectedDocumentId) {
      void fetchDocument(selectedDocumentId)
      void refreshProcessingState()
    } else {
      setDocumentData(null)
      setPreviewState(initialPreviewState)
      setShowAllChunks(false)
      setShowAllEntities({})
      setHasPreview(null)
      setProcessingState(null)
    }

    const handleProcessed = () => {
      if (selectedDocumentId) {
        void fetchDocument(selectedDocumentId)
        void refreshProcessingState()
      }
    }

    const handleProcessingUpdated = () => {
      void refreshProcessingState()
    }

    if (typeof window !== 'undefined') {
      window.addEventListener('documents:processed', handleProcessed)
      window.addEventListener('documents:processing-updated', handleProcessingUpdated)
    }

    return () => {
      isSubscribed = false
      if (typeof window !== 'undefined') {
        window.removeEventListener('documents:processed', handleProcessed)
        window.removeEventListener('documents:processing-updated', handleProcessingUpdated)
      }
    }
  }, [refreshProcessingState, selectedDocumentId])

  useEffect(() => {
    let isSubscribed = true
    const checkPreview = async () => {
      if (!documentData) return
      try {
        const available = await api.hasDocumentPreview(documentData.id)
        if (isSubscribed) setHasPreview(available)
      } catch (e) {
        if (isSubscribed) setHasPreview(false)
      }
    }

    checkPreview()

    return () => {
      isSubscribed = false
    }
  }, [documentData])

  useEffect(() => {
    if (selectedChunkId !== null && documentData) {
      const chunk = documentData.chunks.find(c => c.index === selectedChunkId || c.id === selectedChunkId)
      if (chunk) {
        setExpandedChunks(prev => ({ ...prev, [chunk.id]: true }))
      }
      clearSelectedChunk()
    }
  }, [selectedChunkId, documentData, clearSelectedChunk])

  useEffect(() => {
    return () => {
      if (previewState.objectUrl) {
        URL.revokeObjectURL(previewState.objectUrl)
      }
    }
  }, [previewState.objectUrl])

  const groupedEntities = useMemo(() => {
    if (!documentData?.entities?.length) {
      return {}
    }

    return documentData.entities.reduce<Record<string, DocumentEntity[]>>((acc, entity) => {
      const typeKey = entity.type || 'Unknown'
      if (!acc[typeKey]) {
        acc[typeKey] = []
      }
      acc[typeKey].push(entity)
      return acc
    }, {})
  }, [documentData?.entities])

  const isMarkdownDocument = useMemo(() => {
    if (!documentData) return false

    const mime = documentData.mime_type?.toLowerCase() || ''
    const markdownMimeTypes = new Set([
      'text/markdown',
      'text/x-markdown',
      'text/md',
      'text/x-md',
      'application/markdown',
    ])

    if (markdownMimeTypes.has(mime)) {
      return true
    }

    const fileName = documentData.file_name?.toLowerCase() || ''
    return fileName.endsWith('.md') || fileName.endsWith('.markdown')
  }, [documentData])

  const toggleChunk = useCallback((chunk: DocumentChunk) => {
    setExpandedChunks((state) => ({
      ...state,
      [chunk.id]: !state[chunk.id],
    }))
  }, [])

  const handleCopyChunk = useCallback(async (chunk: DocumentChunk) => {
    try {
      await navigator.clipboard.writeText(chunk.text)
    } catch (copyError) {
      console.error('Failed to copy chunk', copyError)
    }
  }, [])

  const handleOpenPreview = useCallback(async () => {
    if (!selectedDocumentId) return
    if (previewState.objectUrl) {
      URL.revokeObjectURL(previewState.objectUrl)
    }

    if (documentData?.preview_url) {
      setPreviewState({ url: documentData.preview_url, mimeType: documentData.mime_type, isLoading: false, error: null })
      return
    }

    setPreviewState({ url: null, mimeType: undefined, isLoading: true, error: null, objectUrl: null })

    try {
      const response = await api.getDocumentPreview(selectedDocumentId)

      if ('preview_url' in response) {
        setPreviewState({ url: response.preview_url, mimeType: documentData?.mime_type, isLoading: false, error: null })
        return
      }

      const mimeType = response.headers.get('Content-Type') || undefined
      const blob = await response.blob()
      const objectUrl = URL.createObjectURL(blob)
      setPreviewState({ url: objectUrl, mimeType, isLoading: false, error: null, objectUrl })
    } catch (previewError) {
      const errorMessage = previewError instanceof Error ? previewError.message : 'Unable to load preview'
      setPreviewState({ url: null, mimeType: undefined, isLoading: false, error: errorMessage, objectUrl: null })
      console.error('Failed to load preview', previewError)
    }
  }, [documentData?.mime_type, documentData?.preview_url, previewState.objectUrl, selectedDocumentId])

  const handleClosePreview = useCallback(() => {
    if (previewState.objectUrl) {
      URL.revokeObjectURL(previewState.objectUrl)
    }
    setPreviewState(initialPreviewState)
  }, [previewState.objectUrl])

  const handleReprocessChunks = useCallback(async () => {
    if (!documentData?.id) return
    setIsActionPending(true)
    setActionError(null)
    setActionMessage(null)
    try {
      await api.reprocessDocumentChunks(documentData.id)
      setActionMessage('Chunk processing queued')
      await refreshProcessingState()
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('documents:processing-updated'))
      }
    } catch (reprocessError) {
      setActionError(
        reprocessError instanceof Error
          ? reprocessError.message
          : 'Failed to queue chunk processing'
      )
    } finally {
      setIsActionPending(false)
    }
  }, [documentData?.id, refreshProcessingState])

  const handleReprocessEntities = useCallback(async () => {
    if (!documentData?.id) return
    setIsActionPending(true)
    setActionError(null)
    setActionMessage(null)
    try {
      await api.reprocessDocumentEntities(documentData.id)
      setActionMessage('Entity extraction queued')
      await refreshProcessingState()
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('documents:processing-updated'))
      }
    } catch (reprocessError) {
      setActionError(
        reprocessError instanceof Error
          ? reprocessError.message
          : 'Failed to queue entity extraction'
      )
    } finally {
      setIsActionPending(false)
    }
  }, [documentData?.id, refreshProcessingState])

  const handleGenerateSummary = useCallback(async () => {
    if (!documentData?.id) return
    setIsActionPending(true)
    setActionError(null)
    setActionMessage(null)
    try {
      await api.generateDocumentSummary(documentData.id)
      setActionMessage('Summary generated successfully')
      // Refresh document data
      const data = await api.getDocument(documentData.id)
      setDocumentData(data)
    } catch (summaryError) {
      setActionError(
        summaryError instanceof Error
          ? summaryError.message
          : 'Failed to generate summary'
      )
    } finally {
      setIsActionPending(false)
    }
  }, [documentData?.id])

  const handleStartEditHashtags = useCallback(() => {
    setNewHashtagInput('')
    setIsEditingHashtags(true)
  }, [])

  const saveHashtags = useCallback(async (newHashtags: string[]) => {
    if (!documentData?.id) return
    setActionError(null)
    try {
      await api.updateDocumentHashtags(documentData.id, newHashtags)
      // Refresh document data
      const data = await api.getDocument(documentData.id)
      setDocumentData(data)
    } catch (updateError) {
      setActionError(
        updateError instanceof Error
          ? updateError.message
          : 'Failed to update hashtags'
      )
    }
  }, [documentData?.id])

  const handleRemoveHashtag = useCallback(async (tagToRemove: string) => {
    if (!documentData?.hashtags) return
    const newHashtags = documentData.hashtags.filter(tag => tag !== tagToRemove)
    await saveHashtags(newHashtags)
  }, [documentData?.hashtags, saveHashtags])

  const handleAddHashtag = useCallback(async () => {
    let trimmed = newHashtagInput.trim()
    if (!trimmed) return
    
    // Automatically add # prefix if not present
    if (!trimmed.startsWith('#')) {
      trimmed = '#' + trimmed
    }
    
    const currentHashtags = documentData?.hashtags || []
    if (currentHashtags.includes(trimmed)) {
      setNewHashtagInput('')
      return
    }
    
    const newHashtags = [...currentHashtags, trimmed]
    await saveHashtags(newHashtags)
    setNewHashtagInput('')
  }, [newHashtagInput, documentData?.hashtags, saveHashtags])

  const handleHashtagInputKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleAddHashtag()
    }
  }, [handleAddHashtag])

  useEffect(() => {
    if (!actionMessage) return
    const timer = window.setTimeout(() => setActionMessage(null), 4000)
    return () => window.clearTimeout(timer)
  }, [actionMessage])

  useEffect(() => {
    if (!actionError) return
    const timer = window.setTimeout(() => setActionError(null), 5000)
    return () => window.clearTimeout(timer)
  }, [actionError])

  useEffect(() => {
    const isProcessingNow = Boolean(processingState?.is_processing)
    const wasProcessing = prevProcessingRef.current
    prevProcessingRef.current = isProcessingNow

    if (isProcessingNow && selectedDocumentId) {
      const intervalId = window.setInterval(() => {
        // Poll both processing state and document data to refresh summary
        void refreshProcessingState()
        void (async () => {
          try {
            const updatedDoc = await api.getDocument(selectedDocumentId)
            setDocumentData(updatedDoc)
          } catch (error) {
            console.error('Failed to refresh document during processing:', error)
          }
        })()
      }, 2000)
      return () => window.clearInterval(intervalId)
    }

    if (wasProcessing && !isProcessingNow) {
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('documents:processed'))
      }
    }
  }, [processingState?.is_processing, refreshProcessingState, selectedDocumentId])

  const disableManualActions = isActionPending || Boolean(processingState?.is_processing)
  const manualActionTitle = processingState?.is_processing
    ? 'Processing is already running for another document'
    : undefined

  const handleRelatedDocumentClick = useCallback(
    (doc: RelatedDocument) => {
      if (!doc.id) return
      selectDocument(doc.id)
    },
    [selectDocument]
  )

  const toggleShowAllEntities = useCallback((type: string) => {
    setShowAllEntities((state) => ({
      ...state,
      [type]: !state[type],
    }))
  }, [])

  if (!selectedDocumentId) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-secondary-600">
        <DocumentTextIcon className="w-16 h-16 text-secondary-300 mb-4" />
        <p className="text-base font-medium">Select a document to view its details.</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="border-b border-secondary-200 bg-white px-6 py-4 flex items-center gap-4">
        <button
          type="button"
          onClick={clearSelectedDocument}
          className="button-secondary flex items-center gap-2 text-sm"
        >
          <ArrowLeftIcon className="w-4 h-4" />
          Back to chat
        </button>
        <div className="flex items-center gap-3">
          <DocumentTextIcon className="w-6 h-6 text-primary-500" />
          <div>
            <h2 className="text-lg font-semibold text-secondary-900">
              {documentData?.title || documentData?.original_filename || documentData?.file_name || 'Unnamed document'}
            </h2>
            <p className="text-xs text-secondary-500">
              {documentData?.document_type ? documentData.document_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) : 'Unknown type'}
            </p>
          </div>
        </div>
        <div className="ml-auto flex items-center gap-2">
          {documentData?.uploaded_at && (
            <span className="text-xs text-secondary-500">
              Uploaded {new Date(documentData.uploaded_at).toLocaleString()}
            </span>
          )}
          {(hasPreview === null ? documentData?.preview_url : hasPreview) && (
            <button
              type="button"
              onClick={handleOpenPreview}
              className="button-primary text-sm flex items-center gap-2"
              disabled={previewState.isLoading}
            >
              <MagnifyingGlassIcon className="w-4 h-4" />
              {previewState.isLoading ? 'Loading preview…' : 'Open preview'}
            </button>
          )}
          {previewState.error && (
            <span className="text-xs text-red-600">{previewState.error}</span>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-6 bg-secondary-50">
        {isLoading && (
          <div className="flex flex-col items-center justify-center py-20 text-secondary-500">
            <Squares2X2Icon className="w-10 h-10 animate-spin" />
            <p className="mt-3 text-sm">Loading document metadata…</p>
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg p-4 flex items-start gap-2">
            <ExclamationCircleIcon className="w-5 h-5 flex-shrink-0" />
            <div>
              <p className="font-medium">Failed to load document</p>
              <p className="text-sm">{error}</p>
            </div>
          </div>
        )}

        {!isLoading && !error && documentData && (
          <div className="space-y-6">
            {actionMessage && (
              <div className="bg-green-50 border border-green-200 text-green-700 rounded-lg px-4 py-3 text-sm">
                {actionMessage}
              </div>
            )}
            {actionError && (
              <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg px-4 py-3 text-sm">
                {actionError}
              </div>
            )}
            {documentData.metadata?.processing_status === 'staged' && (
              <div className="bg-blue-50 border border-blue-200 text-blue-700 rounded-lg px-4 py-3 text-sm">
                <p className="font-medium mb-1">Document ready to process</p>
                <p>This document has been uploaded but not yet processed. Go to the <strong>Upload</strong> tab in the sidebar and click <strong>Process All</strong> to begin processing.</p>
              </div>
            )}
            <section className="bg-white rounded-lg shadow-sm border border-secondary-200 p-5">
              <h3 className="text-sm font-semibold text-secondary-900 mb-4">Overview</h3>
              <dl className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
                <div>
                  <dt className="text-secondary-500">File name</dt>
                  <dd className="text-secondary-900">{documentData.original_filename || documentData.file_name || 'Unknown'}</dd>
                </div>
                <div>
                  <dt className="text-secondary-500">Mime type</dt>
                  <dd className="text-secondary-900">{documentData.mime_type || 'Unknown'}</dd>
                </div>
                <div>
                  <dt className="text-secondary-500">Uploader</dt>
                  <dd className="text-secondary-900">
                    {documentData.uploader?.name || documentData.uploader?.id || 'Unknown'}
                  </dd>
                </div>
                <div>
                  <dt className="text-secondary-500">Uploaded at</dt>
                  <dd className="text-secondary-900">
                    {documentData.uploaded_at
                      ? new Date(documentData.uploaded_at).toLocaleString()
                      : 'Unknown'}
                  </dd>
                </div>
                <div>
                  <dt className="text-secondary-500">Chunk count</dt>
                  <dd className="text-secondary-900">{documentData.chunks.length}</dd>
                </div>
                <div>
                  <dt className="text-secondary-500">Preview available</dt>
                  <dd className="text-secondary-900">
                    {hasPreview === null
                      ? (documentData.preview_url ? 'Yes' : 'No')
                      : hasPreview
                      ? 'Yes'
                      : 'No'}
                  </dd>
                </div>
              </dl>
              <div className="mt-4 pt-4 border-t border-secondary-200">
                <dt className="text-secondary-500 text-sm mb-2">Tags</dt>
                <div 
                  className="flex flex-wrap gap-2 cursor-pointer p-2 -m-2 rounded hover:bg-secondary-50 transition-colors"
                  onClick={handleStartEditHashtags}
                >
                  {!isEditingHashtags ? (
                    <>
                      {(documentData.hashtags || []).map((tag, index) => (
                        <span
                          key={index}
                          className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary-100 text-primary-800"
                        >
                          {tag}
                        </span>
                      ))}
                      {(!documentData.hashtags || documentData.hashtags.length === 0) && (
                        <span className="text-sm text-secondary-400">Click to add tags</span>
                      )}
                    </>
                  ) : (
                    <div 
                      className="flex flex-wrap gap-2 w-full"
                      onClick={(e) => e.stopPropagation()}
                    >
                      {(documentData.hashtags || []).map((tag, index) => (
                        <span
                          key={index}
                          className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary-100 text-primary-800"
                        >
                          {tag}
                          <button
                            type="button"
                            onMouseDown={(e) => {
                              e.preventDefault()
                              e.stopPropagation()
                              handleRemoveHashtag(tag)
                            }}
                            className="hover:text-primary-900"
                          >
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </button>
                        </span>
                      ))}
                      <div className="inline-flex items-center gap-1">
                        <input
                          type="text"
                          value={newHashtagInput}
                          onChange={(e) => setNewHashtagInput(e.target.value)}
                          onKeyDown={handleHashtagInputKeyDown}
                          onBlur={() => setIsEditingHashtags(false)}
                          placeholder="Add tag..."
                          className="px-2.5 py-0.5 text-xs border border-secondary-300 rounded-full focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                          autoFocus
                        />
                        <button
                          type="button"
                          onMouseDown={(e) => {
                            e.preventDefault()
                            handleAddHashtag()
                          }}
                          className="text-primary-600 hover:text-primary-700"
                          disabled={!newHashtagInput.trim()}
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </section>

            {(documentData.summary || documentData.chunks.length > 0 || documentData.metadata?.processing_status !== 'staged') && (
              <section className="bg-white rounded-lg shadow-sm border border-secondary-200">
                <header className="flex items-center justify-between px-5 py-4 border-b border-secondary-200">
                  <div className="flex items-center gap-2">
                    <h3 className="text-sm font-semibold text-secondary-900">Summary</h3>
                    {documentData.document_type && (
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-secondary-100 text-secondary-700">
                        {documentData.document_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                      </span>
                    )}
                    {!documentData.summary && documentData.chunks.length > 0 && documentData.metadata?.processing_status !== 'staged' && (
                      <button
                        type="button"
                        onClick={handleGenerateSummary}
                        className="button-primary text-xs"
                        disabled={disableManualActions}
                        title={manualActionTitle}
                      >
                        Generate summary
                      </button>
                    )}
                  </div>
                </header>
                <div className="px-5 py-4">
                  {documentData.summary ? (
                    <div className="prose prose-sm prose-slate max-w-none">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm, remarkBreaks]}
                      >
                        {documentData.summary}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <p className="text-sm text-secondary-500">No summary generated yet.</p>
                  )}
                </div>
              </section>
            )}

            <section className="bg-white rounded-lg shadow-sm border border-secondary-200">
              <header className="flex items-center justify-between px-5 py-4 border-b border-secondary-200">
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-semibold text-secondary-900">Chunks</h3>
                  {documentData.chunks.length === 0 && documentData.metadata?.processing_status !== 'staged' && !processingState?.is_processing && (
                    <button
                      type="button"
                      onClick={handleReprocessChunks}
                      className="button-primary text-xs"
                      disabled={disableManualActions}
                      title={manualActionTitle}
                    >
                      Process chunks
                    </button>
                  )}
                </div>
                <span className="text-xs text-secondary-500">{documentData.chunks.length} entries</span>
              </header>
              <div className="divide-y divide-secondary-200">
                {documentData.chunks.length === 0 ? (
                  <p className="px-5 py-4 text-sm text-secondary-500">No chunks processed yet.</p>
                ) : (
                  <>
                    {(showAllChunks ? documentData.chunks : documentData.chunks.slice(0, CHUNKS_LIMIT)).map((chunk: DocumentChunk) => {
                      const expanded = expandedChunks[chunk.id]
                      const firstLine = (chunk.text || '').split(/\r?\n/)[0] || ''
                      const previewLine = firstLine.trim() || 'No preview available'
                      return (
                        <article key={chunk.id} className="px-5 py-4">
                          <div className="flex flex-wrap items-center gap-3">
                            <div className="flex-1 min-w-0">
                              <p className="text-xs text-secondary-500">
                                Chunk {typeof chunk.index === 'number' ? chunk.index + 1 : chunk.id}
                              </p>
                              <div className="relative overflow-hidden pr-8">
                                {/* Allow preview line to wrap and break long words instead of truncating */}
                                <p className="text-sm font-medium text-secondary-900 break-words break-all">
                                  {previewLine}
                                </p>
                              </div>
                            </div>
                            <div className="flex items-center gap-2 flex-shrink-0">
                              <button
                                type="button"
                                onClick={() => handleCopyChunk(chunk)}
                                className="button-ghost text-xs flex items-center gap-1"
                              >
                                <ClipboardIcon className="w-4 h-4" />
                                Copy
                              </button>
                              <button
                                type="button"
                                onClick={() => toggleChunk(chunk)}
                                className="button-secondary text-xs flex items-center gap-1"
                              >
                                {expanded ? (
                                  <ChevronUpIcon className="w-4 h-4" />
                                ) : (
                                  <ChevronDownIcon className="w-4 h-4" />
                                )}
                              </button>
                            </div>
                          </div>
                          <AnimatePresence>
                            {expanded && (
                              <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                transition={{ duration: 0.3, ease: 'easeInOut' }}
                                className="mt-3 text-sm text-secondary-800 leading-relaxed overflow-hidden border-t border-secondary-200 pt-3"
                              >
                                {isMarkdownDocument ? (
                                  <ReactMarkdown
                                    className="prose prose-sm prose-slate max-w-none break-words"
                                    remarkPlugins={[remarkGfm, remarkBreaks]}
                                  >
                                    {chunk.text || ''}
                                  </ReactMarkdown>
                                ) : (
                                  // Use whitespace-pre-wrap to preserve newlines but allow breaking long words
                                  <p className="whitespace-pre-wrap break-words">{chunk.text ?? ''}</p>
                                )}
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </article>
                      )
                    })}
                    {documentData.chunks.length > CHUNKS_LIMIT && (
                      <div className="px-5 py-4 border-t border-secondary-200">
                        <button
                          type="button"
                          onClick={() => setShowAllChunks(!showAllChunks)}
                          className="button-secondary text-sm flex items-center gap-2"
                        >
                          {showAllChunks ? 'Show Less' : `Show ${documentData.chunks.length - CHUNKS_LIMIT} more Chunks`}
                        </button>
                      </div>
                    )}
                  </>
                )}
              </div>
            </section>

            <section className="bg-white rounded-lg shadow-sm border border-secondary-200">
              <header className="flex items-center justify-between px-5 py-4 border-b border-secondary-200">
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-semibold text-secondary-900">Entities</h3>
                  {documentData.entities.length === 0 && documentData.chunks.length > 0 && documentData.metadata?.processing_status !== 'staged' && (
                    <button
                      type="button"
                      onClick={handleReprocessEntities}
                      className="button-primary text-xs"
                      disabled={disableManualActions}
                      title={manualActionTitle}
                    >
                      Process entities
                    </button>
                  )}
                </div>
                <span className="text-xs text-secondary-500">
                  {documentData.entities.length > 0
                    ? `${documentData.entities.length} total`
                    : 'No entities'}
                </span>
              </header>
              {documentData.entities.length === 0 ? (
                <p className="px-5 py-4 text-sm text-secondary-500">No entities extracted yet.</p>
              ) : (
                <div className="divide-y divide-secondary-100">
                  {Object.entries(groupedEntities).map(([type, entities]) => {
                    const showAll = showAllEntities[type] || false
                    const displayedEntities = showAll ? entities : entities.slice(0, ENTITIES_PER_TYPE_LIMIT)
                    const hasMore = entities.length > ENTITIES_PER_TYPE_LIMIT

                    return (
                      <div key={type} className="px-5 py-4 space-y-2">
                        <h4 className="text-xs font-semibold text-secondary-500 uppercase tracking-wide">
                          {type}
                        </h4>
                        <ul className="grid grid-cols-1 md:grid-cols-2 gap-2">
                          {displayedEntities.map((entity) => (
                            <li key={`${type}-${entity.text}`} className="border border-secondary-200 rounded-lg px-3 py-2">
                              <p className="text-sm font-medium text-secondary-900">{entity.text}</p>
                              <p className="text-xs text-secondary-500">
                                Count: {entity.count ?? '—'} · Positions: {entity.positions?.join(', ') || '—'}
                              </p>
                            </li>
                          ))}
                        </ul>
                        {hasMore && (
                          <button
                            type="button"
                            onClick={() => toggleShowAllEntities(type)}
                            className="button-secondary text-xs mt-2"
                          >
                            {showAll ? 'Show Less' : `Show ${entities.length - ENTITIES_PER_TYPE_LIMIT} more ${type.toLowerCase()}s`}
                          </button>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
            </section>

            {documentData.quality_scores && (
              <section className="bg-white rounded-lg shadow-sm border border-secondary-200 p-5">
                <h3 className="text-sm font-semibold text-secondary-900 mb-3">Quality scores</h3>
                <pre className="bg-secondary-900 text-secondary-50 text-xs rounded-lg p-4 overflow-auto">
                  {JSON.stringify(documentData.quality_scores, null, 2)}
                </pre>
              </section>
            )}

            {documentData.related_documents && documentData.related_documents.length > 0 && (
              <section className="bg-white rounded-lg shadow-sm border border-secondary-200 p-5">
                <h3 className="text-sm font-semibold text-secondary-900 mb-3">Related documents</h3>
                <ul className="space-y-2">
                  {documentData.related_documents.map((doc) => (
                    <li key={doc.id} className="flex items-center justify-between gap-3 p-3 border border-secondary-200 rounded-lg hover:bg-secondary-50 transition-colors">
                      <div className="flex-1">
                        <p className="text-sm font-medium text-secondary-900">
                          {doc.title || doc.link || doc.id}
                        </p>
                        {doc.link && (
                          <a
                            href={doc.link}
                            target="_blank"
                            rel="noreferrer"
                            className="text-xs text-primary-600 hover:underline inline-flex items-center gap-1 mt-1"
                          >
                            <span>Open external link</span>
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                            </svg>
                          </a>
                        )}
                      </div>
                      <button
                        type="button"
                        onClick={() => handleRelatedDocumentClick(doc)}
                        className="button-secondary text-xs"
                      >
                        View
                      </button>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {documentData.metadata && (
              <section className="bg-white rounded-lg shadow-sm border border-secondary-200 p-5">
                <h3 className="text-sm font-semibold text-secondary-900 mb-3">Metadata</h3>
                <pre className="bg-secondary-900 text-secondary-50 text-xs rounded-lg p-4 overflow-auto">
                  {JSON.stringify(documentData.metadata, null, 2)}
                </pre>
              </section>
            )}
          </div>
        )}
      </div>

      {previewState.url && (
        <DocumentPreview
          previewUrl={previewState.url}
          mimeType={previewState.mimeType}
          onClose={handleClosePreview}
        />
      )}
    </div>
  )
}
