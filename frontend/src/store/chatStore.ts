import { create } from 'zustand'
import { Message } from '@/types'
import { api } from '@/lib/api'

interface ChatStore {
  messages: Message[]
  sessionId: string
  isHistoryLoading: boolean
  setSessionId: (sessionId: string) => void
  clearChat: () => void
  addMessage: (message: Message) => void
  updateLastMessage: (updater: (previous: Message) => Message) => void
  replaceMessages: (messages: Message[], sessionId: string) => void
  loadSession: (sessionId: string) => Promise<void>
}

export const useChatStore = create<ChatStore>((set, get) => ({
  messages: [],
  sessionId: '',
  isHistoryLoading: false,
  setSessionId: (sessionId) => set({ sessionId }),
  clearChat: () => set({ messages: [], sessionId: '' }),
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
