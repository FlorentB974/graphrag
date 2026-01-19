'use client'

import { useState, useMemo } from 'react'
import { Source } from '@/types'
import { ChevronDownIcon, ChevronUpIcon, DocumentTextIcon } from '@heroicons/react/24/outline'
import { motion, AnimatePresence } from 'framer-motion'
import { useChatStore } from '@/store/chatStore'

interface SourcesListProps {
  sources: Source[]
}

interface GroupedSource {
  documentName: string
  documentId: string
  chunks: Source[]
  avgSimilarity: number
  entityCount: number
}

export default function SourcesList({ sources }: SourcesListProps) {
  const [expanded, setExpanded] = useState(false)
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null)
  const selectDocumentChunk = useChatStore((state) => state.selectDocumentChunk)

  // Group sources by document
  const groupedSources = useMemo(() => {
    const groups = new Map<string, GroupedSource>()

    sources.forEach((source) => {
      // Use original_filename if available, fall back to document_name
      const docName = source.original_filename || source.document_name || source.filename || 'Unknown Document'
      const docKey = docName // Use document name as the key

      if (!groups.has(docKey)) {
        // Find the first source with a valid document_id to use as the group's documentId
        const docId = source.document_id || docKey
        groups.set(docKey, {
          documentName: docName,
          documentId: docId,
          chunks: [],
          avgSimilarity: 0,
          entityCount: 0,
        })
      }

      const group = groups.get(docKey)!
      
      // If this source has a document_id and the group doesn't have one yet, use it
      if (source.document_id && group.documentId === docKey) {
        group.documentId = source.document_id
      }
      
      group.chunks.push(source)
      
      // Count entities (handle entity sources)
      if (source.entity_name) {
        group.entityCount++
      }
    })

    // Calculate average similarity for each document
    groups.forEach((group) => {
      const validSimilarities = group.chunks
        .map((c) => c.similarity || c.relevance_score || 0)
        .filter((s) => !isNaN(s) && s > 0)
      
      if (validSimilarities.length > 0) {
        group.avgSimilarity = validSimilarities.reduce((a, b) => a + b, 0) / validSimilarities.length
      } else {
        group.avgSimilarity = 0
      }
    })

    // Sort by average similarity
    return Array.from(groups.values()).sort((a, b) => b.avgSimilarity - a.avgSimilarity)
  }, [sources])

  const visibleDocs = expanded ? groupedSources : groupedSources.slice(0, 3)

  return (
    <div className="space-y-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center text-sm font-medium text-secondary-700 dark:text-secondary-300 hover:text-secondary-900 dark:hover:text-secondary-100"
      >
        <DocumentTextIcon className="w-4 h-4 mr-1" />
        Sources ({groupedSources.length} {groupedSources.length === 1 ? 'document' : 'documents'})
        {expanded ? (
          <ChevronUpIcon className="w-4 h-4 ml-1" />
        ) : (
          <ChevronDownIcon className="w-4 h-4 ml-1" />
        )}
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            key="sources-expanded"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{
              duration: 0.3,
              ease: [0.4, 0, 0.2, 1],
              opacity: { duration: 0.2 }
            }}
            className="space-y-2 overflow-hidden"
          >
          {visibleDocs.map((doc, index) => (
            <div
              key={doc.documentId}
              className="bg-secondary-50 dark:bg-secondary-800 rounded-lg p-3 cursor-pointer hover:bg-secondary-100 dark:hover:bg-secondary-700 transition-colors"
              onClick={() => setSelectedDoc(selectedDoc === doc.documentId ? null : doc.documentId)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap md:flex-nowrap">
                    <div className="min-w-0 flex-1">
                      <span
                        className="block truncate text-sm font-medium text-secondary-900 dark:text-secondary-50"
                        title={doc.documentName}
                      >
                        {doc.documentName}
                      </span>
                    </div>
                    {doc.avgSimilarity > 0 && (
                      <span className="shrink-0 text-xs px-2 py-0.5 bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded">
                        {(doc.avgSimilarity * 100).toFixed(0)}% match
                      </span>
                    )}
                    <span className="shrink-0 text-xs text-secondary-600 dark:text-secondary-400">
                      {doc.chunks.length} {doc.chunks.length === 1 ? 'chunk' : 'chunks'}
                    </span>
                    {doc.entityCount > 0 && (
                      <span className="shrink-0 text-xs px-2 py-0.5 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded">
                        {doc.entityCount} {doc.entityCount === 1 ? 'entity' : 'entities'}
                      </span>
                    )}
                  </div>
                </div>
                <div>
                  {selectedDoc === doc.documentId ? (
                    <ChevronUpIcon className="w-4 h-4 text-secondary-500 dark:text-secondary-400" />
                  ) : (
                    <ChevronDownIcon className="w-4 h-4 text-secondary-500 dark:text-secondary-400" />
                  )}
                </div>
              </div>

              <AnimatePresence>
                {selectedDoc === doc.documentId && (
                  <motion.div
                    key={`doc-${doc.documentId}`}
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{
                      duration: 0.3,
                      ease: [0.4, 0, 0.2, 1],
                      opacity: { duration: 0.2 }
                    }}
                    className="mt-3 pt-3 border-t border-secondary-200 dark:border-secondary-700 space-y-2 overflow-hidden"
                  >
                  {doc.chunks.map((chunk, chunkIndex) => {
                    const similarity = chunk.similarity || chunk.relevance_score || 0
                    const handleChunkClick = () => {
                      if (chunk.chunk_index !== undefined) {
                        // Regular chunk - navigate to specific chunk
                        selectDocumentChunk(doc.documentId, chunk.chunk_index)
                      } else {
                        // Entity source - just navigate to the document
                        selectDocumentChunk(doc.documentId, 0)
                      }
                    }
                    return (
                      <div key={chunkIndex} className="bg-white dark:bg-secondary-700 rounded p-2 text-sm cursor-pointer hover:bg-secondary-50 dark:hover:bg-secondary-600 transition-colors" onClick={handleChunkClick}>
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center space-x-2 flex-wrap">
                            {chunk.entity_name ? (
                              <span className="text-xs font-medium text-purple-700 dark:text-purple-300">
                                🏷️ Entity: {chunk.entity_name}
                              </span>
                            ) : (
                              chunk.chunk_index !== undefined && (
                                <span className="text-xs text-secondary-600 dark:text-secondary-400">
                                  Section {chunk.chunk_index + 1}
                                </span>
                              )
                            )}
                            {!isNaN(similarity) && similarity > 0 && (
                              <span className="text-xs text-secondary-600 dark:text-secondary-400">
                                {(similarity * 100).toFixed(0)}% relevance
                              </span>
                            )}
                          </div>
                        </div>
                        <p className="text-xs text-secondary-700 dark:text-secondary-300 whitespace-pre-wrap break-words line-clamp-3">
                          {chunk.content.substring(0, 200)}
                          {chunk.content.length > 200 && '...'}
                        </p>
                        {chunk.contained_entities && chunk.contained_entities.length > 0 && (
                          <div className="mt-2 flex flex-wrap gap-1">
                            {chunk.contained_entities.slice(0, 5).map((entity) => (
                              <span
                                key={entity}
                                className="text-xs px-1.5 py-0.5 bg-secondary-200 dark:bg-secondary-600 text-secondary-700 dark:text-secondary-200 rounded"
                              >
                                {entity}
                              </span>
                            ))}
                            {chunk.contained_entities.length > 5 && (
                              <span className="text-xs text-secondary-600 dark:text-secondary-400">
                                +{chunk.contained_entities.length - 5} more
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </motion.div>
              )}
              </AnimatePresence>
            </div>
          ))}

          {!expanded && groupedSources.length > 3 && (
            <button
              onClick={() => setExpanded(true)}
              className="text-sm text-primary-600 hover:text-primary-700"
            >
              Show {groupedSources.length - 3} more documents
            </button>
          )}
        </motion.div>
      )}
      </AnimatePresence>
    </div>
  )
}

