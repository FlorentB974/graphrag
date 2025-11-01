'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircleIcon, XCircleIcon, XMarkIcon } from '@heroicons/react/24/outline'

export interface ToastProps {
  id: string
  type: 'success' | 'error'
  message: string
  description?: string
  onDismiss: (id: string) => void
}

export default function Toast({ id, type, message, description, onDismiss }: ToastProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 50, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className={`flex items-start gap-3 rounded-lg border px-4 py-3 shadow-lg max-w-md ${
        type === 'success'
          ? 'border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/30'
          : 'border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/30'
      }`}
    >
      {type === 'success' ? (
        <CheckCircleIcon className="h-5 w-5 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
      ) : (
        <XCircleIcon className="h-5 w-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
      )}
      <div className="flex-1 min-w-0">
        <p className={`text-sm font-medium ${
          type === 'success' ? 'text-green-900 dark:text-green-300' : 'text-red-900 dark:text-red-300'
        }`}>
          {message}
        </p>
        {description && (
          <p className={`text-xs mt-1 ${
            type === 'success' ? 'text-green-700 dark:text-green-400' : 'text-red-700 dark:text-red-400'
          }`}>
            {description}
          </p>
        )}
      </div>
      <button
        onClick={() => onDismiss(id)}
        className={`flex-shrink-0 ${
          type === 'success' ? 'text-green-600 dark:text-green-400 hover:text-green-800 dark:hover:text-green-300' : 'text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300'
        }`}
      >
        <XMarkIcon className="h-4 w-4" />
      </button>
    </motion.div>
  )
}
