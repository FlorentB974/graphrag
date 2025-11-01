'use client'

import { useState, useCallback, useEffect } from 'react'
import { AnimatePresence } from 'framer-motion'
import Toast, { ToastProps } from './Toast'

interface ToastData {
  id: string
  type: 'success' | 'error'
  message: string
  description?: string
  duration?: number
}

let toastCounter = 0
const toastCallbacks: Set<(toast: ToastData) => void> = new Set()

export function showToast(
  type: 'success' | 'error',
  message: string,
  description?: string,
  duration = 5000
) {
  const toast: ToastData = {
    id: `toast-${++toastCounter}`,
    type,
    message,
    description,
    duration,
  }
  toastCallbacks.forEach(cb => cb(toast))
}

export default function ToastContainer() {
  const [toasts, setToasts] = useState<ToastData[]>([])

  const dismissToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  const addToast = useCallback((toast: ToastData) => {
    setToasts(prev => [...prev, toast])
    
    // Auto-dismiss after duration
    if (toast.duration && toast.duration > 0) {
      setTimeout(() => {
        dismissToast(toast.id)
      }, toast.duration)
    }
  }, [dismissToast])

  useEffect(() => {
    toastCallbacks.add(addToast)
    return () => {
      toastCallbacks.delete(addToast)
    }
  }, [addToast])

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
      <AnimatePresence>
        {toasts.map(toast => (
          <Toast
            key={toast.id}
            {...toast}
            onDismiss={dismissToast}
          />
        ))}
      </AnimatePresence>
    </div>
  )
}
