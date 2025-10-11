"use client"

/* eslint-disable @next/next/no-img-element */

import { useEffect } from 'react'
import { XMarkIcon, ArrowTopRightOnSquareIcon } from '@heroicons/react/24/outline'

type DocumentPreviewProps = {
  previewUrl: string
  mimeType?: string
  onClose?: () => void
}

const isPdf = (mimeType?: string) => mimeType?.includes('pdf') ?? false
const isImage = (mimeType?: string) =>
  mimeType?.startsWith('image/') ?? false

export default function DocumentPreview({ previewUrl, mimeType, onClose }: DocumentPreviewProps) {
  useEffect(() => {
    const handleEsc = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose?.()
      }
    }

    document.addEventListener('keydown', handleEsc)
    return () => document.removeEventListener('keydown', handleEsc)
  }, [onClose])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4">
      <div className="relative w-full max-w-5xl bg-white rounded-xl shadow-2xl overflow-hidden">
        <div className="flex items-center justify-between border-b border-secondary-200 px-4 py-3 bg-secondary-50">
          <div>
            <p className="text-sm font-medium text-secondary-900">Document Preview</p>
            <p className="text-xs text-secondary-500">Press Escape to close</p>
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

        <div className="h-[70vh] bg-secondary-100 p-4 overflow-auto">
          {isPdf(mimeType) ? (
            <iframe
              src={previewUrl}
              title="Document preview"
              className="w-full h-full rounded-lg border border-secondary-200"
            />
          ) : isImage(mimeType) ? (
            <div className="flex items-center justify-center h-full">
              <img
                src={previewUrl}
                alt="Document preview"
                className="max-h-full max-w-full rounded-lg shadow"
              />
            </div>
          ) : (
            <iframe
              src={previewUrl}
              title="Document preview"
              className="w-full h-full rounded-lg border border-secondary-200 bg-white"
            />
          )}
        </div>
      </div>
    </div>
  )
}
