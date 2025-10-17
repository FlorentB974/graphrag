'use client'

import { useState, useRef, useEffect } from 'react'
import { PaperAirplaneIcon } from '@heroicons/react/24/solid'
import { StopIcon, DocumentArrowUpIcon } from '@heroicons/react/24/outline'
import { api } from '@/lib/api'

interface ChatInputProps {
  onSend: (message: string) => void
  disabled?: boolean
  isStreaming?: boolean
  onStop?: () => void
  onFileUploaded?: () => void
  userMessages?: string[]
}

export default function ChatInput({
  onSend,
  onStop,
  disabled,
  isStreaming,
  onFileUploaded,
  userMessages = [],
}: ChatInputProps) {
  const [input, setInput] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const [uploadingFile, setUploadingFile] = useState(false)
  const [historyIndex, setHistoryIndex] = useState(-1)
  const [savedInput, setSavedInput] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

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

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (isStreaming) {
      return
    }
    if (input.trim() && !disabled) {
      onSend(input)
      setInput('')
      setHistoryIndex(-1)
      setSavedInput('')
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
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
      }

      // Notify parent component that files were uploaded
      if (onFileUploaded) {
        onFileUploaded()
      }
    } catch (error) {
      console.error('Failed to stage file:', error)
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

          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents..."
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

