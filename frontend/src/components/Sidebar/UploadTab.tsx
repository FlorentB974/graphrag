'use client'

import { useState } from 'react'
import { api } from '@/lib/api'
import { showToast } from '@/components/Toast/ToastContainer'
import {
  CloudArrowUpIcon,
  XCircleIcon,
} from '@heroicons/react/24/outline'

export default function UploadTab() {
  const [isDragging, setIsDragging] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null)

  const handleFiles = async (files: FileList | File[]) => {
    setUploadError(null)
    setUploadSuccess(null)

    const fileArray = Array.from(files)

    for (const file of fileArray) {
      try {
        const result = await api.stageFile(file)

        if (result.status === 'queued' || result.status === 'staged') {
          // Notify other tabs that a document was uploaded
          if (typeof window !== 'undefined') {
            window.dispatchEvent(new CustomEvent('documents:uploaded'))
          }
          
          // Show toast notification
          showToast('success', `${file.name} uploaded`, 'Document queued for processing')
          
          setUploadSuccess(`${file.name} uploaded and queued for processing`)
          // Clear success message after 3 seconds
          setTimeout(() => setUploadSuccess(null), 3000)
        } else {
          const errorMsg = result.error || 'Failed to upload file'
          setUploadError(errorMsg)
          showToast('error', `Failed to upload ${file.name}`, errorMsg)
        }
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : 'Failed to upload file'
        setUploadError(errorMsg)
        showToast('error', `Failed to upload ${file.name}`, errorMsg)
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

  return (
    <div className="space-y-4">
      {/* Drop Zone */}
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          isDragging
            ? 'border-primary-500 bg-primary-50'
            : 'border-secondary-300 hover:border-primary-500'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          type="file"
          id="file-upload"
          className="hidden"
          onChange={handleFileUpload}
          accept=".pdf,.txt,.md,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.jpg,.jpeg,.png"
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
            PDF, DOCX, TXT, MD, PPT, XLS, Images
          </span>
        </label>
      </div>

      {/* Success Message */}
      {uploadSuccess && (
        <div className="p-4 rounded-lg flex items-start bg-green-50 text-green-800">
          <CloudArrowUpIcon className="w-5 h-5 mr-2 flex-shrink-0" />
          <p className="text-sm">{uploadSuccess}</p>
        </div>
      )}

      {/* Error Message */}
      {uploadError && (
        <div className="p-4 rounded-lg flex items-start bg-red-50 text-red-800">
          <XCircleIcon className="w-5 h-5 mr-2 flex-shrink-0" />
          <p className="text-sm">{uploadError}</p>
        </div>
      )}

      {/* Info Section */}
      <div className="text-xs text-secondary-600 space-y-1">
        <p className="font-medium">How it works:</p>
        <ul className="list-disc list-inside space-y-0.5 ml-2">
          <li>Upload documents here or in the chat</li>
          <li>Documents are automatically queued for processing</li>
          <li>Check the Database tab to monitor progress</li>
          <li>Processed documents are available for chat</li>
        </ul>
        <p className="font-medium mt-3">Supported formats:</p>
        <ul className="list-disc list-inside space-y-0.5 ml-2">
          <li>PDF documents</li>
          <li>Word documents (.doc, .docx)</li>
          <li>PowerPoint (.ppt, .pptx)</li>
          <li>Excel (.xls, .xlsx)</li>
          <li>Text files (.txt, .md)</li>
          <li>Images (.jpg, .png) - with OCR</li>
        </ul>
      </div>
    </div>
  )
}
