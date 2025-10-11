"use client"

import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  ArrowLeftIcon,
  DocumentTextIcon,
  Squares2X2Icon,
  MagnifyingGlassIcon,
  ClipboardIcon,
  ExclamationCircleIcon,
} from '@heroicons/react/24/outline'
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/solid'
import { api } from '@/lib/api'
import type {
  DocumentChunk,
  DocumentDetails,
  DocumentEntity,
  RelatedDocument,
} from '@/types'
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
  const clearSelectedDocument = useChatStore((state) => state.clearSelectedDocument)
  const selectDocument = useChatStore((state) => state.selectDocument)

  const [documentData, setDocumentData] = useState<DocumentDetails | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [expandedChunks, setExpandedChunks] = useState<Record<string | number, boolean>>({})
  const [previewState, setPreviewState] = useState<PreviewState>(initialPreviewState)

  useEffect(() => {
    let isSubscribed = true

    const fetchDocument = async (documentId: string) => {
      setIsLoading(true)
      setError(null)
      setDocumentData(null)
      setPreviewState(initialPreviewState)

      try {
        const data = await api.getDocument(documentId)
        if (isSubscribed) {
          setDocumentData(data)
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
      fetchDocument(selectedDocumentId)
    } else {
      setDocumentData(null)
      setPreviewState(initialPreviewState)
    }

    return () => {
      isSubscribed = false
    }
  }, [selectedDocumentId])

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
      setPreviewState({ url: null, mimeType: undefined, isLoading: false, error: 'Unable to load preview', objectUrl: null })
      console.error('Failed to load preview', previewError)
    }
  }, [documentData?.mime_type, documentData?.preview_url, previewState.objectUrl, selectedDocumentId])

  const handleClosePreview = useCallback(() => {
    if (previewState.objectUrl) {
      URL.revokeObjectURL(previewState.objectUrl)
    }
    setPreviewState(initialPreviewState)
  }, [previewState.objectUrl])

  const handleRelatedDocumentClick = useCallback(
    (doc: RelatedDocument) => {
      if (!doc.id) return
      selectDocument(doc.id)
    },
    [selectDocument]
  )

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
              {documentData?.title || documentData?.file_name || 'Unnamed document'}
            </h2>
            <p className="text-xs text-secondary-500">
              {documentData?.file_name} · {documentData?.mime_type || 'Unknown type'}
            </p>
          </div>
        </div>
        <div className="ml-auto flex items-center gap-2">
          {documentData?.uploaded_at && (
            <span className="text-xs text-secondary-500">
              Uploaded {new Date(documentData.uploaded_at).toLocaleString()}
            </span>
          )}
          <button
            type="button"
            onClick={handleOpenPreview}
            className="button-primary text-sm flex items-center gap-2"
            disabled={previewState.isLoading}
          >
            <MagnifyingGlassIcon className="w-4 h-4" />
            {previewState.isLoading ? 'Loading preview…' : 'Open preview'}
          </button>
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
            <section className="bg-white rounded-lg shadow-sm border border-secondary-200 p-5">
              <h3 className="text-sm font-semibold text-secondary-900 mb-4">Overview</h3>
              <dl className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
                <div>
                  <dt className="text-secondary-500">File name</dt>
                  <dd className="text-secondary-900">{documentData.file_name || 'Unknown'}</dd>
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
                  <dd className="text-secondary-900">{documentData.preview_url ? 'Yes' : 'No'}</dd>
                </div>
              </dl>
            </section>

            <section className="bg-white rounded-lg shadow-sm border border-secondary-200">
              <header className="flex items-center justify-between px-5 py-4 border-b border-secondary-200">
                <h3 className="text-sm font-semibold text-secondary-900">Chunks</h3>
                <span className="text-xs text-secondary-500">{documentData.chunks.length} entries</span>
              </header>
              <div className="divide-y divide-secondary-200">
                {documentData.chunks.map((chunk: DocumentChunk) => {
                  const expanded = expandedChunks[chunk.id]
                  return (
                    <article key={chunk.id} className="px-5 py-4">
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <p className="text-xs text-secondary-500">
                            Chunk {typeof chunk.index === 'number' ? chunk.index + 1 : chunk.id}
                          </p>
                          <p className="text-xs text-secondary-400">
                            Offset: {chunk.offset ?? '—'} · Score: {chunk.score ?? '—'}
                          </p>
                        </div>
                        <div className="flex items-center gap-2">
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
                            {expanded ? 'Collapse' : 'Expand'}
                          </button>
                        </div>
                      </div>
                      {expanded && (
                        <p className="mt-3 text-sm text-secondary-800 whitespace-pre-wrap leading-relaxed">
                          {chunk.text}
                        </p>
                      )}
                    </article>
                  )
                })}
              </div>
            </section>

            <section className="bg-white rounded-lg shadow-sm border border-secondary-200">
              <header className="flex items-center justify-between px-5 py-4 border-b border-secondary-200">
                <h3 className="text-sm font-semibold text-secondary-900">Entities</h3>
                <span className="text-xs text-secondary-500">
                  {documentData.entities.length > 0
                    ? `${documentData.entities.length} total`
                    : 'No entities'}
                </span>
              </header>
              {documentData.entities.length === 0 ? (
                <p className="px-5 py-4 text-sm text-secondary-500">No entities extracted.</p>
              ) : (
                <div className="divide-y divide-secondary-100">
                  {Object.entries(groupedEntities).map(([type, entities]) => (
                    <div key={type} className="px-5 py-4 space-y-2">
                      <h4 className="text-xs font-semibold text-secondary-500 uppercase tracking-wide">
                        {type}
                      </h4>
                      <ul className="grid grid-cols-1 md:grid-cols-2 gap-2">
                        {entities.map((entity) => (
                          <li key={`${type}-${entity.text}`} className="border border-secondary-200 rounded-lg px-3 py-2">
                            <p className="text-sm font-medium text-secondary-900">{entity.text}</p>
                            <p className="text-xs text-secondary-500">
                              Count: {entity.count ?? '—'} · Positions: {entity.positions?.join(', ') || '—'}
                            </p>
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
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
                    <li key={doc.id} className="flex items-center justify-between gap-3">
                      <div>
                        <p className="text-sm font-medium text-secondary-900">
                          {doc.title || doc.link || doc.id}
                        </p>
                        {doc.link && (
                          <a
                            href={doc.link}
                            target="_blank"
                            rel="noreferrer"
                            className="text-xs text-primary-600 hover:underline"
                          >
                            Open reference
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
