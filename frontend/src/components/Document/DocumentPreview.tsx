"use client"

/* eslint-disable @next/next/no-img-element */

import { useEffect, useState } from 'react'
import { XMarkIcon, ArrowTopRightOnSquareIcon } from '@heroicons/react/24/outline'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'

type DocumentPreviewProps = {
  previewUrl: string
  mimeType?: string
  onClose?: () => void
}

const isPdf = (mimeType?: string) => mimeType?.includes('pdf') ?? false
const isImage = (mimeType?: string) => mimeType?.startsWith('image/') ?? false
const isMarkdown = (mimeType?: string) => 
  (mimeType?.includes('markdown') || mimeType?.includes('text/markdown')) ?? false
const isOfficeDoc = (mimeType?: string) => {
  if (!mimeType) return false
  return (
    mimeType.includes('officedocument') || // Microsoft Office formats
    mimeType.includes('ms-word') ||
    mimeType.includes('ms-excel') ||
    mimeType.includes('ms-powerpoint') ||
    mimeType.includes('application/vnd.ms-') ||
    mimeType === 'application/msword' ||
    mimeType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
    mimeType === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' ||
    mimeType === 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
  )
}

export default function DocumentPreview({ previewUrl, mimeType, onClose }: DocumentPreviewProps) {
  const [markdownContent, setMarkdownContent] = useState<string>('')
  const [isLoadingMarkdown, setIsLoadingMarkdown] = useState(false)

  useEffect(() => {
    const handleEsc = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose?.()
      }
    }

    document.addEventListener('keydown', handleEsc)
    return () => document.removeEventListener('keydown', handleEsc)
  }, [onClose])

  // Fetch markdown content if it's a markdown file
  useEffect(() => {
    if (isMarkdown(mimeType)) {
      setIsLoadingMarkdown(true)
      fetch(previewUrl)
        .then(res => res.text())
        .then(text => {
          setMarkdownContent(text)
          setIsLoadingMarkdown(false)
        })
        .catch(err => {
          console.error('Failed to load markdown:', err)
          setMarkdownContent('Failed to load markdown content')
          setIsLoadingMarkdown(false)
        })
    }
  }, [previewUrl, mimeType])

  // Generate Office Online viewer URL for Office documents
  const getOfficeViewerUrl = (url: string) => {
    // Use Microsoft Office Online viewer
    return `https://view.officeapps.live.com/op/embed.aspx?src=${encodeURIComponent(url)}`
  }

  const renderPreviewContent = () => {
    if (isPdf(mimeType)) {
      return (
        <iframe
          src={previewUrl}
          title="Document preview"
          className="w-full h-full rounded-lg border border-secondary-200"
        />
      )
    }

    if (isImage(mimeType)) {
      return (
        <div className="flex items-center justify-center h-full">
          <img
            src={previewUrl}
            alt="Document preview"
            className="max-h-full max-w-full rounded-lg shadow"
          />
        </div>
      )
    }

    if (isMarkdown(mimeType)) {
      if (isLoadingMarkdown) {
        return (
          <div className="flex items-center justify-center h-full">
            <div className="text-secondary-600 dark:text-secondary-400">Loading markdown...</div>
          </div>
        )
      }
      return (
        <div className="prose prose-secondary max-w-none p-6 bg-white dark:bg-secondary-800 rounded-lg">
          <ReactMarkdown 
            remarkPlugins={[remarkGfm, remarkBreaks]}
            components={{
              // Customize code blocks
              code({ className, children, ...props }) {
                const isInline = !className?.includes('language-')
                return isInline ? (
                  <code className="px-1.5 py-0.5 rounded bg-secondary-100 dark:bg-secondary-700 text-secondary-800 dark:text-secondary-200 text-sm font-mono" {...props}>
                    {children}
                  </code>
                ) : (
                  <code className={`block p-4 rounded bg-secondary-900 text-secondary-100 text-sm font-mono overflow-x-auto ${className}`} {...props}>
                    {children}
                  </code>
                )
              },
              // Customize links to open in new tab
              a({ children, href, ...props }) {
                return (
                  <a href={href} target="_blank" rel="noopener noreferrer" className="text-primary-600 hover:text-primary-700 underline" {...props}>
                    {children}
                  </a>
                )
              }
            }}
          >
            {markdownContent}
          </ReactMarkdown>
        </div>
      )
    }

    if (isOfficeDoc(mimeType)) {
      // For Office documents, we'll try multiple approaches
      // First attempt: Microsoft Office Online viewer (requires publicly accessible URL)
      // Second attempt: Direct iframe (may work for some browsers/formats)
      
      // Check if URL is absolute (publicly accessible)
      const isAbsoluteUrl = previewUrl.startsWith('http://') || previewUrl.startsWith('https://')
      
      if (isAbsoluteUrl) {
        // Use Office Online viewer for publicly accessible files
        return (
          <div className="w-full h-full space-y-4">
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-sm text-yellow-800">
              <p className="font-medium">Office Document Viewer</p>
              <p className="text-xs mt-1">
                Using Microsoft Office Online viewer. If the document doesn&apos;t load, 
                try opening it in a new tab or downloading it.
              </p>
            </div>
            <iframe
              src={getOfficeViewerUrl(previewUrl)}
              title="Office document preview"
              className="w-full h-[calc(100%-5rem)] rounded-lg border border-secondary-200"
            />
          </div>
        )
      } else {
        // For local files, show a message and download option
        return (
          <div className="flex flex-col items-center justify-center h-full space-y-4 p-8">
            <div className="text-center space-y-2">
              <div className="text-4xl">📄</div>
              <h3 className="text-lg font-medium text-secondary-900">
                Microsoft Office Document
              </h3>
              <p className="text-sm text-secondary-600 dark:text-secondary-400 max-w-md">
                Office documents (Word, Excel, PowerPoint) cannot be previewed directly in the browser 
                for local files. Please download the file to view it.
              </p>
            </div>
            <a
              href={previewUrl}
              download
              className="button-primary flex items-center gap-2"
            >
              <ArrowTopRightOnSquareIcon className="w-5 h-5" />
              Download Document
            </a>
          </div>
        )
      }
    }

    // Default fallback
    return (
      <iframe
        src={previewUrl}
        title="Document preview"
        className="w-full h-full rounded-lg border border-secondary-200 dark:border-secondary-600 bg-white dark:bg-secondary-800"
      />
    )
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4">
      <div className="relative w-full max-w-5xl bg-white dark:bg-secondary-800 rounded-xl shadow-2xl overflow-hidden">
        <div className="flex items-center justify-between border-b border-secondary-200 dark:border-secondary-700 px-4 py-3 bg-secondary-50 dark:bg-secondary-700">
          <div>
            <p className="text-sm font-medium text-secondary-900 dark:text-secondary-50">Document Preview</p>
            <p className="text-xs text-secondary-500 dark:text-secondary-400">Press Escape to close</p>
          </div>
          <div className="flex items-center gap-2">
            <a
              href={previewUrl}
              target="_blank"
              rel="noreferrer"
              className="button-ghost text-xs flex items-center gap-1"
            >
              <ArrowTopRightOnSquareIcon className="w-4 h-4" />
              Open in new tab
            </a>
            <button
              type="button"
              onClick={onClose}
              className="button-secondary text-xs flex items-center gap-1"
            >
              <XMarkIcon className="w-4 h-4" />
              Close
            </button>
          </div>
        </div>

        <div className="h-[70vh] bg-secondary-100 dark:bg-secondary-900 p-4 overflow-auto">
          {renderPreviewContent()}
        </div>
      </div>
    </div>
  )
}
