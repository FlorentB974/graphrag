'use client'

import { useChatStore } from '@/store/chatStore'
import { ExclamationTriangleIcon } from '@heroicons/react/24/solid'

export default function ConnectionStatus() {
  const isConnected = useChatStore((state) => state.isConnected)
  const isStreamingRequest = useChatStore((state) => state.isStreamingRequest)

  // Don't show disconnection warning if we're actively streaming
  // The active SSE connection proves the server is responsive
  if (isConnected || isStreamingRequest) {
    return null
  }

  return (
    <div className="flex items-center justify-center gap-3 px-4 py-3 bg-red-50 dark:bg-red-900/30 border-b border-red-200 dark:border-red-800">
      <ExclamationTriangleIcon className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0" />
      <div className="flex-1">
        <p className="text-sm font-medium text-red-800 dark:text-red-200">
          Unable to connect to server
        </p>
        <p className="text-xs text-red-700 dark:text-red-300">
          Chat is currently unavailable. Please check your connection and try again.
        </p>
      </div>
      <div className="flex-shrink-0">
        <div className="inline-flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-red-600 dark:bg-red-400 animate-pulse" />
          <span className="text-xs font-medium text-red-600 dark:text-red-400">Offline</span>
        </div>
      </div>
    </div>
  )
}
