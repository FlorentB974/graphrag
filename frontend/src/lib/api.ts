import type { DocumentDetails, DocumentChunk, DocumentTextPayload } from '@/types'
import type { ProcessingProgressResponse } from '@/types/upload'

// Dynamically determine API URL for remote client support
const getApiUrl = (): string => {
  // Use Next.js proxy if enabled
  if (process.env.NEXT_PUBLIC_USE_PROXY === 'true') {
    return ''
  }
  
  // Use explicit API URL if set
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL
  }
  
  // Auto-detect from browser location
  if (typeof window !== 'undefined') {
    const protocol = window.location.protocol
    const host = window.location.hostname
    return `${protocol}//${host}:8000`
  }
  
  // SSR fallback
  return 'http://localhost:8000'
}

const API_URL = getApiUrl()

// Default timeout values (in milliseconds)
const DEFAULT_TIMEOUT = 30000 // 30 seconds for most operations
const LONG_TIMEOUT = 120000 // 2 minutes for upload/processing operations
const SHORT_TIMEOUT = 10000 // 10 seconds for quick operations

/**
 * Custom error class for timeout errors
 */
export class TimeoutError extends Error {
  constructor(message: string = 'Request timed out') {
    super(message)
    this.name = 'TimeoutError'
  }
}

/**
 * Fetch wrapper with configurable timeout
 */
async function fetchWithTimeout(
  url: string,
  options: RequestInit & { timeout?: number } = {}
): Promise<Response> {
  const { timeout = DEFAULT_TIMEOUT, signal: externalSignal, ...fetchOptions } = options
  
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), timeout)
  
  // Combine external signal with timeout signal
  const combinedSignal = externalSignal
    ? combineAbortSignals(externalSignal, controller.signal)
    : controller.signal
  
  try {
    const response = await fetch(url, {
      ...fetchOptions,
      signal: combinedSignal,
    })
    clearTimeout(timeoutId)
    return response
  } catch (error) {
    clearTimeout(timeoutId)
    if (error instanceof Error && error.name === 'AbortError') {
      // Check if it was a timeout or external abort
      if (controller.signal.aborted && !externalSignal?.aborted) {
        throw new TimeoutError(`Request to ${url} timed out after ${timeout}ms`)
      }
    }
    throw error
  }
}

/**
 * Combines multiple AbortSignals into one
 */
function combineAbortSignals(...signals: AbortSignal[]): AbortSignal {
  const controller = new AbortController()
  
  for (const signal of signals) {
    if (signal.aborted) {
      controller.abort()
      break
    }
    signal.addEventListener('abort', () => controller.abort(), { once: true })
  }
  
  return controller.signal
}

export const api = {
  // Chat endpoints
  async sendMessage(
    data: {
      message: string
      session_id?: string
      model?: string
      retrieval_mode?: string
      top_k?: number
      temperature?: number
      top_p?: number
      use_multi_hop?: boolean
      stream?: boolean
      context_documents?: string[]
      context_document_labels?: string[]
      context_hashtags?: string[]
      llm_overrides?: {
        model?: string
        temperature?: number
        top_k?: number
      }
    },
    options?: { signal?: AbortSignal }
  ) {
    const response = await fetch(`${API_URL}/api/chat/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
      signal: options?.signal,
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }

    return response
  },

  // History endpoints
  async getHistory() {
    const response = await fetchWithTimeout(`${API_URL}/api/history/sessions`, {
      timeout: DEFAULT_TIMEOUT,
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getConversation(sessionId: string) {
    const response = await fetchWithTimeout(`${API_URL}/api/history/${sessionId}`, {
      timeout: DEFAULT_TIMEOUT,
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async deleteConversation(sessionId: string) {
    const response = await fetchWithTimeout(`${API_URL}/api/history/${sessionId}`, {
      method: 'DELETE',
      timeout: DEFAULT_TIMEOUT,
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async clearHistory() {
    const response = await fetchWithTimeout(`${API_URL}/api/history/clear`, {
      method: 'POST',
      timeout: DEFAULT_TIMEOUT,
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  // Database endpoints
  async getStats() {
    const response = await fetchWithTimeout(`${API_URL}/api/database/stats`, {
      timeout: DEFAULT_TIMEOUT,
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async uploadFile(file: File) {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetchWithTimeout(`${API_URL}/api/database/upload`, {
      method: 'POST',
      body: formData,
      timeout: LONG_TIMEOUT,
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async stageFile(file: File) {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetchWithTimeout(`${API_URL}/api/database/stage`, {
      method: 'POST',
      body: formData,
      timeout: LONG_TIMEOUT,
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getStagedDocuments() {
    const response = await fetchWithTimeout(`${API_URL}/api/database/staged`, {
      timeout: DEFAULT_TIMEOUT,
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async deleteStagedDocument(fileId: string) {
    const response = await fetchWithTimeout(`${API_URL}/api/database/staged/${fileId}`, {
      method: 'DELETE',
      timeout: DEFAULT_TIMEOUT,
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async processDocuments(fileIds: string[]) {
    const response = await fetchWithTimeout(`${API_URL}/api/database/process`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ file_ids: fileIds }),
      timeout: LONG_TIMEOUT,
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getProcessingProgress(fileId?: string): Promise<ProcessingProgressResponse> {
    const url = fileId
      ? `${API_URL}/api/database/progress/${fileId}`
      : `${API_URL}/api/database/progress`
    const response = await fetchWithTimeout(url, {
      timeout: SHORT_TIMEOUT,
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async deleteDocument(documentId: string) {
    const response = await fetchWithTimeout(`${API_URL}/api/database/documents/${documentId}`, {
      method: 'DELETE',
      timeout: LONG_TIMEOUT,
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async clearDatabase() {
    const response = await fetchWithTimeout(`${API_URL}/api/database/clear`, {
      method: 'POST',
      timeout: LONG_TIMEOUT,
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async reprocessDocumentChunks(documentId: string) {
    const response = await fetchWithTimeout(
      `${API_URL}/api/database/documents/${documentId}/process/chunks`,
      {
        method: 'POST',
        timeout: LONG_TIMEOUT,
      }
    )
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async reprocessDocumentEntities(documentId: string) {
    const response = await fetchWithTimeout(
      `${API_URL}/api/database/documents/${documentId}/process/entities`,
      {
        method: 'POST',
        timeout: LONG_TIMEOUT,
      }
    )
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getDocuments() {
    const response = await fetchWithTimeout(`${API_URL}/api/database/documents`, {
      timeout: DEFAULT_TIMEOUT,
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getHashtags() {
    const response = await fetchWithTimeout(`${API_URL}/api/database/hashtags`, {
      timeout: DEFAULT_TIMEOUT,
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getDocument(documentId: string): Promise<DocumentDetails> {
    const response = await fetchWithTimeout(`${API_URL}/api/documents/${documentId}`, {
      timeout: DEFAULT_TIMEOUT,
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getDocumentChunks(documentId: string): Promise<DocumentChunk[]> {
    const response = await fetchWithTimeout(`${API_URL}/api/documents/${documentId}/chunks`, {
      timeout: DEFAULT_TIMEOUT,
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }

    const payload = await response.json()
    return Array.isArray(payload?.chunks) ? payload.chunks : []
  },

  async getDocumentText(documentId: string): Promise<DocumentTextPayload> {
    const response = await fetchWithTimeout(`${API_URL}/api/documents/${documentId}/text`, {
      timeout: DEFAULT_TIMEOUT,
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async getDocumentPreview(
    documentId: string
  ): Promise<{ preview_url: string } | Response> {
    const response = await fetchWithTimeout(`${API_URL}/api/documents/${documentId}/preview`, {
      redirect: 'follow',
      timeout: LONG_TIMEOUT,
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }

    const contentType = response.headers.get('Content-Type') || ''
    if (contentType.includes('application/json')) {
      return response.json()
    }

    return response
  },

  async generateDocumentSummary(documentId: string) {
    const response = await fetchWithTimeout(`${API_URL}/api/documents/${documentId}/generate-summary`, {
      method: 'POST',
      timeout: LONG_TIMEOUT,
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }

    return response.json()
  },

  async updateDocumentHashtags(documentId: string, hashtags: string[]) {
    const response = await fetchWithTimeout(`${API_URL}/api/documents/${documentId}/hashtags`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ hashtags }),
      timeout: DEFAULT_TIMEOUT,
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }

    return response.json()
  },

  async hasDocumentPreview(documentId: string): Promise<boolean> {
    // Try a HEAD request first to avoid downloading the full file.
    try {
      const headResp = await fetchWithTimeout(`${API_URL}/api/documents/${documentId}/preview`, {
        method: 'HEAD',
        redirect: 'follow',
        timeout: SHORT_TIMEOUT,
      })

      if (headResp.ok) return true

      // Some servers may not accept HEAD. Try a lightweight GET requesting only the first byte.
      if (headResp.status === 405) {
        const getResp = await fetchWithTimeout(`${API_URL}/api/documents/${documentId}/preview`, {
          method: 'GET',
          redirect: 'follow',
          headers: {
            Range: 'bytes=0-0',
          },
          timeout: SHORT_TIMEOUT,
        })
        return getResp.ok || getResp.status === 206
      }

      return false
    } catch (err) {
      return false
    }
  },

  async getSettings() {
    const response = await fetchWithTimeout(`${API_URL}/api/health`, {
      timeout: SHORT_TIMEOUT,
    })
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`)
    }
    return response.json()
  },

  async checkHealth(signal?: AbortSignal): Promise<boolean> {
    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 5000)
      
      if (signal) {
        signal.addEventListener('abort', () => controller.abort())
      }
      
      const response = await fetch(`${API_URL}/api/health`, {
        method: 'GET',
        signal: controller.signal,
      })
      
      clearTimeout(timeoutId)
      return response.ok
    } catch (error) {
      return false
    }
  },

}
