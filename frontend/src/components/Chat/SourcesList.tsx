'use client'

import { useState, useMemo } from 'react'
import { Source } from '@/types'
import { ChevronDownIcon, ChevronUpIcon, DocumentTextIcon } from '@heroicons/react/24/outline'

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

  // Group sources by document
  const groupedSources = useMemo(() => {
    const groups = new Map<string, GroupedSource>()

    sources.forEach((source) => {
      const docKey = source.document_id || source.document_name || source.filename || 'Unknown'
      const docName = source.document_name || source.filename || 'Unknown Document'

      if (!groups.has(docKey)) {
        groups.set(docKey, {
          documentName: docName,
          documentId: docKey,
          chunks: [],
          avgSimilarity: 0,
          entityCount: 0,
        })
      }

      const group = groups.get(docKey)!
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
        className="flex items-center text-sm font-medium text-secondary-700 hover:text-secondary-900"
      >
        <DocumentTextIcon className="w-4 h-4 mr-1" />
        Sources ({groupedSources.length} {groupedSources.length === 1 ? 'document' : 'documents'})
        {expanded ? (
          <ChevronUpIcon className="w-4 h-4 ml-1" />
        ) : (
          <ChevronDownIcon className="w-4 h-4 ml-1" />
        )}
      </button>

      {expanded && (
        <div className="space-y-2">
          {visibleDocs.map((doc, index) => (
            <div
              key={doc.documentId}
              className="bg-secondary-50 rounded-lg p-3 cursor-pointer hover:bg-secondary-100 transition-colors"
              onClick={() => setSelectedDoc(selectedDoc === doc.documentId ? null : doc.documentId)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap md:flex-nowrap">
                    <div className="min-w-0 flex-1">
                      <span
                        className="block truncate text-sm font-medium text-secondary-900"
                        title={doc.documentName}
                      >
                        {doc.documentName}
                      </span>
                    </div>
                    {doc.avgSimilarity > 0 && (
                      <span className="shrink-0 text-xs px-2 py-0.5 bg-primary-100 text-primary-700 rounded">
                        {(doc.avgSimilarity * 100).toFixed(0)}% match
                      </span>
                    )}
                    <span className="shrink-0 text-xs text-secondary-600">
                      {doc.chunks.length} {doc.chunks.length === 1 ? 'chunk' : 'chunks'}
                    </span>
                    {doc.entityCount > 0 && (
                      <span className="shrink-0 text-xs px-2 py-0.5 bg-purple-100 text-purple-700 rounded">
                        {doc.entityCount} {doc.entityCount === 1 ? 'entity' : 'entities'}
                      </span>
                    )}
                  </div>
                </div>
                <div>
                  {selectedDoc === doc.documentId ? (
                    <ChevronUpIcon className="w-4 h-4 text-secondary-500" />
                  ) : (
                    <ChevronDownIcon className="w-4 h-4 text-secondary-500" />
                  )}
                </div>
              </div>

              {selectedDoc === doc.documentId && (
                <div className="mt-3 pt-3 border-t border-secondary-200 space-y-2">
                  {doc.chunks.map((chunk, chunkIndex) => {
                    const similarity = chunk.similarity || chunk.relevance_score || 0
                    return (
                      <div key={chunkIndex} className="bg-white rounded p-2 text-sm">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center space-x-2 flex-wrap">
                            {chunk.entity_name ? (
                              <span className="text-xs font-medium text-purple-700">
                                🏷️ Entity: {chunk.entity_name}
                              </span>
                            ) : (
                              chunk.chunk_index !== undefined && (
                                <span className="text-xs text-secondary-600">
                                  Section {chunk.chunk_index + 1}
                                </span>
                              )
                            )}
                            {!isNaN(similarity) && similarity > 0 && (
                              <span className="text-xs text-secondary-600">
                                {(similarity * 100).toFixed(0)}% relevance
                              </span>
                            )}
                          </div>
                        </div>
                        <p className="text-xs text-secondary-700 whitespace-pre-wrap break-words line-clamp-3">
                          {chunk.content.substring(0, 200)}
                          {chunk.content.length > 200 && '...'}
                        </p>
                        {chunk.contained_entities && chunk.contained_entities.length > 0 && (
                          <div className="mt-2 flex flex-wrap gap-1">
                            {chunk.contained_entities.slice(0, 5).map((entity, i) => (
                              <span
                                key={i}
                                className="text-xs px-1.5 py-0.5 bg-secondary-200 text-secondary-700 rounded"
                              >
                                {entity}
                              </span>
                            ))}
                            {chunk.contained_entities.length > 5 && (
                              <span className="text-xs text-secondary-600">
                                +{chunk.contained_entities.length - 5} more
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
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
        </div>
      )}
    </div>
  )
}

