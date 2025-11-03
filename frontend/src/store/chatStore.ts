import { create } from 'zustand'
import { Message } from '@/types'
import { api } from '@/lib/api'

type ActiveView = 'chat' | 'document'

interface ChatStore {
  messages: Message[]
  sessionId: string
  isHistoryLoading: boolean
  // Increment this key to notify history UI to refresh
  historyRefreshKey: number
  activeView: ActiveView
  selectedDocumentId: string | null
  selectedChunkId: string | number | null
  isConnected: boolean
  setIsConnected: (connected: boolean) => void
  notifyHistoryRefresh: () => void
  setSessionId: (sessionId: string) => void
  setActiveView: (view: ActiveView) => void
  clearChat: () => void
  selectDocument: (documentId: string) => void
  selectDocumentChunk: (documentId: string, chunkId: string | number) => void
  clearSelectedDocument: () => void
  clearSelectedChunk: () => void
  addMessage: (message: Message) => void
  updateLastMessage: (updater: (previous: Message) => Message) => void
  replaceMessages: (messages: Message[], sessionId: string) => void
  loadSession: (sessionId: string) => Promise<void>
}

export const useChatStore = create<ChatStore>((set, get) => ({
  messages: [],
  sessionId: '',
  isHistoryLoading: false,
  historyRefreshKey: 0,
  activeView: 'chat',
  selectedDocumentId: null,
  selectedChunkId: null,
  isConnected: true,
  setIsConnected: (connected) => set({ isConnected: connected }),
  setSessionId: (sessionId) => set({ sessionId }),
  setActiveView: (view) => set({ activeView: view }),
  notifyHistoryRefresh: () => set((state) => ({ historyRefreshKey: state.historyRefreshKey + 1 })),
  clearChat: () => {
    // Clear UI state and notify history to refresh
    set({ messages: [], sessionId: '', activeView: 'chat', selectedDocumentId: null, selectedChunkId: null })
    get().notifyHistoryRefresh()
  },
  selectDocument: (documentId) =>
    set({ selectedDocumentId: documentId, activeView: 'document', selectedChunkId: null }),
  selectDocumentChunk: (documentId, chunkId) =>
    set({ selectedDocumentId: documentId, activeView: 'document', selectedChunkId: chunkId }),
  clearSelectedDocument: () => set({ selectedDocumentId: null, selectedChunkId: null, activeView: 'chat' }),
  clearSelectedChunk: () => set({ selectedChunkId: null }),
  addMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message],
    })),
  updateLastMessage: (updater) =>
    set((state) => {
      if (state.messages.length === 0) {
        return state
      }
      const updatedMessages = [...state.messages]
      const lastIndex = updatedMessages.length - 1
      updatedMessages[lastIndex] = updater(updatedMessages[lastIndex])
      return { messages: updatedMessages }
    }),
  replaceMessages: (messages, sessionId) =>
    set({
      messages,
      sessionId,
    }),
  loadSession: async (sessionId: string) => {
    if (!sessionId) return
    set({ isHistoryLoading: true })
    try {
      const conversation = await api.getConversation(sessionId)
      const mappedMessages: Message[] = (conversation.messages || []).map(
        (message: any) => ({
          role: message.role,
          content: message.content,
          timestamp: message.timestamp,
          sources: message.sources || [],
          quality_score: message.quality_score || undefined,
          follow_up_questions: message.follow_up_questions || undefined,
          isStreaming: false,
          context_documents: message.context_documents || undefined,
          context_document_labels: message.context_document_labels || undefined,
          context_hashtags: message.context_hashtags || undefined,
        })
      )

      set({
        messages: mappedMessages,
        sessionId: conversation.session_id || sessionId,
      })
    } catch (error) {
      throw error
    } finally {
      set({ isHistoryLoading: false })
    }
  },
}))
