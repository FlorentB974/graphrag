'use client'

import { useState, useRef, useEffect, useMemo } from 'react'
import { PaperAirplaneIcon } from '@heroicons/react/24/solid'
import { StopIcon, DocumentArrowUpIcon, XMarkIcon } from '@heroicons/react/24/outline'
import { api } from '@/lib/api'
import { DocumentSummary } from '@/types'
import { showToast } from '@/components/Toast/ToastContainer'

type SelectedDocMap = Record<string, { filename: string }>

interface ChatInputProps {
  onSend: (message: string, contextDocuments: string[], contextDocumentLabels: string[], contextHashtags?: string[]) => void
  disabled?: boolean
  isStreaming?: boolean
  onStop?: () => void
  userMessages?: string[]
}

export default function ChatInput({
  onSend,
  onStop,
  disabled,
  isStreaming,
  userMessages = [],
}: ChatInputProps) {
  const [input, setInput] = useState('')
  const [documents, setDocuments] = useState<DocumentSummary[]>([])
  const [documentsLoaded, setDocumentsLoaded] = useState(false)
  const [hashtags, setHashtags] = useState<string[]>([])
  const [hashtagsLoaded, setHashtagsLoaded] = useState(false)
  const [selectedDocs, setSelectedDocs] = useState<SelectedDocMap>({})
  const [selectedHashtags, setSelectedHashtags] = useState<string[]>([])
  const [mentionState, setMentionState] = useState<{ start: number; query: string; type: 'document' | 'hashtag' } | null>(null)
  const [mentionIndex, setMentionIndex] = useState(0)
  const [showMentionList, setShowMentionList] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [uploadingFile, setUploadingFile] = useState(false)
  const [historyIndex, setHistoryIndex] = useState(-1)
  const [savedInput, setSavedInput] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    let isMounted = true

    const fetchDocuments = async () => {
      try {
        const response = await api.getDocuments()
        const docs = Array.isArray(response?.documents) ? response.documents : []
        if (isMounted) {
          setDocuments(docs)
        }
      } catch (error) {
        console.error('Failed to fetch documents:', error)
      } finally {
        if (isMounted) {
          setDocumentsLoaded(true)
        }
      }
    }

    const fetchHashtags = async () => {
      try {
        const response = await api.getHashtags()
        const tags = Array.isArray(response?.hashtags) ? response.hashtags : []
        if (isMounted) {
          setHashtags(tags)
        }
      } catch (error) {
        console.error('Failed to fetch hashtags:', error)
      } finally {
        if (isMounted) {
          setHashtagsLoaded(true)
        }
      }
    }

    fetchDocuments()
    fetchHashtags()

    return () => {
      isMounted = false
    }
  }, [])

  // Auto-resize textarea based on content
  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = `${textarea.scrollHeight}px`
    }
  }

  // Adjust height when input changes
  useEffect(() => {
    adjustTextareaHeight()
  }, [input])

  const selectedDocEntries = useMemo(
    () => Object.entries(selectedDocs),
    [selectedDocs]
  )

  const filteredItems = useMemo(() => {
    if (!mentionState) {
      // Return empty array when there's no active mention
      return []
    }

    const normalized = mentionState.query.toLowerCase()

    if (mentionState.type === 'hashtag') {
      if (!normalized) {
        return hashtags.slice(0, 8)
      }
      return hashtags
        .filter((tag) => tag.toLowerCase().includes(normalized))
        .slice(0, 8)
    } else {
      const available = documents.filter(
        (doc) => !selectedDocs[doc.document_id]
      )
      if (!normalized) {
        return available.slice(0, 8)
      }
      return available
        .filter((doc) => doc.filename.toLowerCase().includes(normalized))
        .slice(0, 8)
    }
  }, [documents, hashtags, mentionState, selectedDocs])

  useEffect(() => {
    if (filteredItems.length === 0) {
      setMentionIndex(0)
      return
    }

    if (mentionIndex >= filteredItems.length) {
      setMentionIndex(0)
    }
  }, [filteredItems, mentionIndex])

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value
    const caret = e.target.selectionStart ?? value.length

    setInput(value)

    const beforeCaret = value.slice(0, caret)
    const atIndex = beforeCaret.lastIndexOf('@')
    const hashIndex = beforeCaret.lastIndexOf('#')

    let nextMention: { start: number; query: string; type: 'document' | 'hashtag' } | null = null

    // Check for '@' (document mention)
    if (atIndex !== -1 && atIndex > hashIndex) {
      const charBefore = atIndex > 0 ? beforeCaret[atIndex - 1] : ''
      const query = beforeCaret.slice(atIndex + 1)
      const hasWhitespace = /\s/.test(query)

      if ((charBefore === '' || /\s/.test(charBefore)) && !hasWhitespace && !query.includes('@')) {
        nextMention = { start: atIndex, query, type: 'document' }
      }
    }
    // Check for '#' (hashtag mention)
    else if (hashIndex !== -1) {
      const charBefore = hashIndex > 0 ? beforeCaret[hashIndex - 1] : ''
      const query = beforeCaret.slice(hashIndex + 1)
      const hasWhitespace = /\s/.test(query)

      if ((charBefore === '' || /\s/.test(charBefore)) && !hasWhitespace && !query.includes('#')) {
        nextMention = { start: hashIndex, query, type: 'hashtag' }
      }
    }

    if (nextMention) {
      setMentionState(nextMention)
      setShowMentionList(true)
      setMentionIndex(0)
    } else {
      setMentionState(null)
      setShowMentionList(false)
    }

  }

  const handleSelectDocument = (doc: DocumentSummary) => {
    if (!mentionState || !textareaRef.current) {
      return
    }

    const value = input
    const caret = textareaRef.current.selectionStart ?? value.length
    const before = value.slice(0, mentionState.start)
    const after = value.slice(caret)
    const newValue = `${before}${after}`

    setInput(newValue)
    setSelectedDocs((prev) => ({
      ...prev,
      [doc.document_id]: {
        filename: doc.filename,
      },
    }))
    setMentionState(null)
    setShowMentionList(false)
    setMentionIndex(0)

    requestAnimationFrame(() => {
      if (textareaRef.current) {
        const nextCaretPosition = before.length
        textareaRef.current.focus()
        textareaRef.current.setSelectionRange(nextCaretPosition, nextCaretPosition)
      }
    })
  }

  const handleSelectHashtag = (hashtag: string) => {
    if (!mentionState || !textareaRef.current) {
      return
    }

    const value = input
    const caret = textareaRef.current.selectionStart ?? value.length
    const before = value.slice(0, mentionState.start)
    const after = value.slice(caret)
    const newValue = `${before}${after}`

    setInput(newValue)
    setSelectedHashtags((prev) => [...prev, hashtag])
    setMentionState(null)
    setShowMentionList(false)
    setMentionIndex(0)

    requestAnimationFrame(() => {
      if (textareaRef.current) {
        const nextCaretPosition = before.length
        textareaRef.current.focus()
        textareaRef.current.setSelectionRange(nextCaretPosition, nextCaretPosition)
      }
    })
  }

  const handleSelectItem = (item: DocumentSummary | string) => {
    if (mentionState?.type === 'hashtag' && typeof item === 'string') {
      handleSelectHashtag(item)
    } else if (typeof item === 'object') {
      handleSelectDocument(item)
    }
  }

  const handleRemoveDoc = (docId: string) => {
    const info = selectedDocs[docId]
    if (!info) {
      return
    }

    setSelectedDocs((prev) => {
      const next = { ...prev }
      delete next[docId]
      return next
    })

    setMentionState(null)
    setShowMentionList(false)
    setMentionIndex(0)
  }

  const handleRemoveHashtag = (hashtag: string) => {
    setSelectedHashtags((prev) => prev.filter((h) => h !== hashtag))
    setMentionState(null)
    setShowMentionList(false)
    setMentionIndex(0)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (isStreaming) {
      return
    }
    if (input.trim() && !disabled) {
      // Start with explicitly selected documents
      const contextDocIds = Object.keys(selectedDocs)
      const contextDocLabels = contextDocIds
        .map((id) => selectedDocs[id]?.filename)
        .filter((label): label is string => Boolean(label))

      // If hashtags are selected, add documents that have those hashtags
      if (selectedHashtags.length > 0) {
        const hashtagDocs = documents.filter((doc) => 
          doc.hashtags && doc.hashtags.some((tag) => selectedHashtags.includes(tag))
        )
        
        // Add these documents to the context if not already included
        hashtagDocs.forEach((doc) => {
          if (!contextDocIds.includes(doc.document_id)) {
            contextDocIds.push(doc.document_id)
            contextDocLabels.push(doc.filename)
          }
        })
      }

      onSend(input, contextDocIds, contextDocLabels, selectedHashtags.length > 0 ? selectedHashtags : undefined)
      setInput('')
      setHistoryIndex(-1)
      setSavedInput('')
      setSelectedDocs({})
      setSelectedHashtags([])
      setMentionState(null)
      setShowMentionList(false)
      setMentionIndex(0)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (showMentionList) {
      if (e.key === 'ArrowDown' && filteredItems.length > 0) {
        e.preventDefault()
        setMentionIndex((prev) => (prev + 1) % filteredItems.length)
        return
      }

      if (e.key === 'ArrowUp' && filteredItems.length > 0) {
        e.preventDefault()
        setMentionIndex((prev) =>
          prev === 0 ? filteredItems.length - 1 : prev - 1
        )
        return
      }

      if (
        (e.key === 'Enter' || e.key === 'Tab') &&
        filteredItems.length > 0 &&
        mentionState
      ) {
        e.preventDefault()
        handleSelectItem(filteredItems[mentionIndex])
        return
      }

      if (e.key === 'Escape') {
        e.preventDefault()
        setShowMentionList(false)
        setMentionState(null)
        return
      }
    }

    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      if (userMessages.length > 0) {
        if (historyIndex === -1) {
          // Save current input before navigating
          setSavedInput(input)
        }
        const newIndex = Math.min(historyIndex + 1, userMessages.length - 1)
        setHistoryIndex(newIndex)
        setInput(userMessages[userMessages.length - 1 - newIndex])
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault()
      if (historyIndex > -1) {
        const newIndex = historyIndex - 1
        setHistoryIndex(newIndex)
        if (newIndex === -1) {
          // Restore saved input
          setInput(savedInput)
        } else {
          setInput(userMessages[userMessages.length - 1 - newIndex])
        }
      }
    }
  }

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

  return (
    <div className="relative">
      <form onSubmit={handleSubmit} className="relative">
        <div
          className={`relative transition-all ${
            isDragging ? 'ring-2 ring-primary-500 rounded-lg' : ''
          }`}
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

          {(selectedDocEntries.length > 0 || selectedHashtags.length > 0) && (
            <div className="mb-2 flex flex-wrap items-center gap-2 pr-24">
              <span className="text-[11px] font-semibold uppercase tracking-wide text-secondary-500">
                Force context
              </span>
              {selectedDocEntries.map(([docId, info]) => (
                <span
                  key={docId}
                  className="inline-flex max-w-full items-center gap-2 rounded-full bg-primary-50 px-3 py-1 text-xs text-primary-700"
                >
                  <span className="truncate" title={info.filename}>
                    {info.filename}
                  </span>
                  <button
                    type="button"
                    onClick={() => handleRemoveDoc(docId)}
                    className="rounded-full p-0.5 text-primary-600 transition hover:bg-primary-100 focus:outline-none focus:ring-1 focus:ring-primary-400"
                    aria-label={`Remove ${info.filename} from forced context`}
                  >
                    <XMarkIcon className="h-3.5 w-3.5" />
                  </button>
                </span>
              ))}
              {selectedHashtags.map((hashtag) => (
                <span
                  key={hashtag}
                  className="inline-flex max-w-full items-center gap-2 rounded-full bg-green-50 px-3 py-1 text-xs text-green-700"
                >
                  <span className="truncate" title={hashtag}>
                    {hashtag.startsWith('#') ? hashtag : `#${hashtag}`}
                  </span>
                  <button
                    type="button"
                    onClick={() => handleRemoveHashtag(hashtag)}
                    className="rounded-full p-0.5 text-green-600 transition hover:bg-green-100 focus:outline-none focus:ring-1 focus:ring-green-400"
                    aria-label={`Remove ${hashtag} from forced context`}
                  >
                    <XMarkIcon className="h-3.5 w-3.5" />
                  </button>
                </span>
              ))}
            </div>
          )}

          {showMentionList && (
            <div
              className="absolute bottom-full left-0 right-24 mb-2 max-h-56 overflow-y-auto rounded-lg border border-secondary-200 bg-white shadow-lg z-20"
              role="listbox"
            >
              {mentionState?.type === 'hashtag' ? (
                !hashtagsLoaded ? (
                  <div className="px-3 py-2 text-sm text-secondary-400">
                    Loading hashtags...
                  </div>
                ) : filteredItems.length === 0 ? (
                  <div className="px-3 py-2 text-sm text-secondary-400">
                    No matching hashtags
                  </div>
                ) : (
                  (filteredItems as string[]).map((tag, idx) => (
                    <button
                      key={tag}
                      type="button"
                      role="option"
                      aria-selected={idx === mentionIndex}
                      onMouseDown={(event) => {
                        event.preventDefault()
                        handleSelectHashtag(tag)
                      }}
                      className={`flex w-full items-center gap-3 px-3 py-2 text-left text-sm transition ${
                        idx === mentionIndex
                          ? 'bg-primary-50 text-primary-700'
                          : 'hover:bg-secondary-100'
                      }`}
                    >
                      <span className="truncate" title={tag}>
                        {tag.startsWith('#') ? tag : `#${tag}`}
                      </span>
                    </button>
                  ))
                )
              ) : (
                !documentsLoaded ? (
                  <div className="px-3 py-2 text-sm text-secondary-400">
                    Loading documents...
                  </div>
                ) : filteredItems.length === 0 ? (
                  <div className="px-3 py-2 text-sm text-secondary-400">
                    No matching documents
                  </div>
                ) : (
                  (filteredItems as DocumentSummary[]).map((doc, idx) => (
                    <button
                      key={doc.document_id}
                      type="button"
                      role="option"
                      aria-selected={idx === mentionIndex}
                      onMouseDown={(event) => {
                        event.preventDefault()
                        handleSelectDocument(doc)
                      }}
                      className={`flex w-full items-center justify-between gap-3 px-3 py-2 text-left text-sm transition ${
                        idx === mentionIndex
                          ? 'bg-primary-50 text-primary-700'
                          : 'hover:bg-secondary-100'
                      }`}
                    >
                      <span className="truncate" title={doc.filename}>
                        {doc.filename}
                      </span>
                      {typeof doc.chunk_count === 'number' && (
                        <span className="text-xs text-secondary-400 whitespace-nowrap">
                          {doc.chunk_count} chunks
                        </span>
                      )}
                    </button>
                  ))
                )
              )}
            </div>
          )}

          <textarea
            ref={textareaRef}
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents... (@ for documents, # for tags)"
            disabled={disabled || isStreaming || uploadingFile}
            rows={1}
            className="input-field pr-24 resize-none overflow-hidden"
          />

          <div
            className="absolute right-3 bottom-3 flex items-center gap-2"
            style={{ transform: 'translateY(-1px)' }}
          >
            {/* File Upload Button */}
            <label
              className={`cursor-pointer p-2 text-secondary-400 hover:text-primary-600 transition-colors ${
                disabled || isStreaming || uploadingFile
                  ? 'opacity-50 pointer-events-none'
                  : ''
              }`}
            >
              <input
                type="file"
                className="hidden"
                onChange={handleFileInput}
                disabled={disabled || isStreaming || uploadingFile}
                accept=".pdf,.txt,.md,.doc,.docx,.ppt,.pptx,.xls,.xlsx"
                multiple
              />
              <DocumentArrowUpIcon className="w-5 h-5" />
            </label>

            {/* Send/Stop Button */}
            {isStreaming ? (
              <button
                type="button"
                onClick={onStop}
                className="inline-flex items-center gap-1 rounded-md bg-rose-500 px-3 py-2 text-xs font-semibold text-white shadow-sm transition hover:bg-rose-600 focus:outline-none focus:ring-2 focus:ring-rose-400 focus:ring-offset-2"
              >
                <StopIcon className="w-4 h-4" />
                Stop
              </button>
            ) : (
              <button
                type="submit"
                disabled={disabled || !input.trim() || uploadingFile}
                className="button-primary p-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <PaperAirplaneIcon className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </form>

      {uploadingFile && (
        <div className="absolute -top-8 left-0 right-0 text-center">
          <span className="text-xs text-secondary-600 bg-white px-2 py-1 rounded shadow-sm">
            Uploading files...
          </span>
        </div>
      )}
    </div>
  )
}

