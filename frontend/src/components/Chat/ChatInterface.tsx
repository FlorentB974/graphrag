'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import { Message } from '@/types'
import { api } from '@/lib/api'
import MessageBubble from './MessageBubble'
import ChatInput from './ChatInput'
import FollowUpQuestions from './FollowUpQuestions'
import LoadingIndicator from './LoadingIndicator'
import { PlusCircleIcon } from '@heroicons/react/24/outline'
import { useChatStore } from '@/store/chatStore'

export default function ChatInterface() {
  const messages = useChatStore((state) => state.messages)
  const sessionId = useChatStore((state) => state.sessionId)
  const addMessage = useChatStore((state) => state.addMessage)
  const updateLastMessage = useChatStore((state) => state.updateLastMessage)
  const setSessionId = useChatStore((state) => state.setSessionId)
  const clearChat = useChatStore((state) => state.clearChat)
  const notifyHistoryRefresh = useChatStore((state) => state.notifyHistoryRefresh)
  const isHistoryLoading = useChatStore((state) => state.isHistoryLoading)
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleStopStreaming = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
  }

  const handleNewChat = useCallback(() => {
    handleStopStreaming()
    // Current session is already saved to history automatically
    // Just clear the UI and reset session ID
    clearChat()
  }, [clearChat])

  // Keyboard shortcut for new chat
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Only trigger if 'n' is pressed and not in an input/textarea
      if (event.key === 'n' && 
          !(document.activeElement instanceof HTMLInputElement) && 
          !(document.activeElement instanceof HTMLTextAreaElement) &&
          !event.ctrlKey && !event.metaKey && !event.altKey) {
        event.preventDefault()
        handleNewChat()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [handleNewChat])

  const handleSendMessage = async (
    message: string,
    contextDocuments: string[],
    contextDocumentLabels: string[],
    contextHashtags?: string[]
  ) => {
    if (!message.trim() || isLoading || isHistoryLoading) return

    // Add user message
    const userMessage: Message = {
      role: 'user',
      content: message,
      timestamp: new Date().toISOString(),
      context_documents: contextDocuments,
      context_document_labels: contextDocumentLabels,
      context_hashtags: contextHashtags,
    }
    addMessage(userMessage)
    setIsLoading(true)

    let accumulatedContent = ''
    let displayedContent = ''
    let sources: any[] = []
    let qualityScore: any = null
    let followUpQuestions: string[] = []
    let newSessionId = sessionId
    let streamCompleted = false
    let animationFrameId: number | null = null
    let contentBuffer: string[] = []
    let effectiveContextDocs = [...contextDocuments]
    const contextDocLabelMap = new Map<string, string>()
    contextDocuments.forEach((id, index) => {
      const label = contextDocumentLabels[index]
      if (label) {
        contextDocLabelMap.set(id, label)
      }
    })
    let effectiveContextDocLabels = [...contextDocumentLabels]

    // Smooth rendering using requestAnimationFrame
    const smoothRender = () => {
      if (contentBuffer.length > 0) {
        // Take a chunk from the buffer
        const chunkSize = Math.ceil(contentBuffer.length / 3) || 1
        const nextChunk = contentBuffer.splice(0, chunkSize).join('')
        displayedContent += nextChunk
        
        updateLastMessage((prev) => ({
          ...prev,
          content: displayedContent,
        }))
      }

      // Continue rendering if there's still content in the buffer or stream is ongoing
      if (contentBuffer.length > 0 || !streamCompleted) {
        animationFrameId = requestAnimationFrame(smoothRender)
      } else if (streamCompleted && accumulatedContent !== displayedContent) {
        // Ensure all content is displayed
        displayedContent = accumulatedContent
        updateLastMessage((prev) => ({
          ...prev,
          content: displayedContent,
          isStreaming: false,
          sources,
          quality_score: qualityScore,
          follow_up_questions: followUpQuestions,
        }))
      }
    }

    try {
      // Add placeholder for assistant message
      const assistantMessage: Message = {
        role: 'assistant',
        content: '',
        isStreaming: true,
      }
      addMessage(assistantMessage)

      const controller = new AbortController()
      abortControllerRef.current = controller

      const response = await api.sendMessage(
        {
          message,
          session_id: sessionId,
          stream: true,
          context_documents: contextDocuments,
        },
        { signal: controller.signal }
      )

      if (!response.body) {
        throw new Error('No response body')
      }

      // Start smooth rendering
      animationFrameId = requestAnimationFrame(smoothRender)

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))

              if (data.type === 'token') {
                accumulatedContent += data.content
                // Add to buffer instead of updating immediately
                contentBuffer.push(data.content)
              } else if (data.type === 'sources') {
                sources = data.content
              } else if (data.type === 'quality_score') {
                qualityScore = data.content
              } else if (data.type === 'follow_ups') {
                followUpQuestions = data.content
              } else if (data.type === 'metadata') {
                newSessionId = data.content.session_id
                if (Array.isArray(data.content.context_documents)) {
                  effectiveContextDocs = data.content.context_documents
                  effectiveContextDocLabels = data.content.context_documents.map(
                    (docId: string) => contextDocLabelMap.get(docId) || docId
                  )
                }
              } else if (data.type === 'done') {
                streamCompleted = true
              } else if (data.type === 'error') {
                throw new Error(data.content)
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e)
            }
          }
        }
      }

      // Mark stream as completed and let animation frame finish
      streamCompleted = true

      // Wait for animation to complete
      await new Promise((resolve) => {
        const checkComplete = () => {
          if (contentBuffer.length === 0 && displayedContent === accumulatedContent) {
            resolve(null)
          } else {
            setTimeout(checkComplete, 50)
          }
        }
        checkComplete()
      })

      // Final update with all metadata
      updateLastMessage((prev) => ({
        ...prev,
        content: accumulatedContent,
        isStreaming: false,
        sources,
        quality_score: qualityScore,
        follow_up_questions: followUpQuestions,
        context_documents: effectiveContextDocs,
        context_document_labels: effectiveContextDocLabels,
      }))

      if (newSessionId && !sessionId) {
        setSessionId(newSessionId)
      }

      notifyHistoryRefresh()
    } catch (error) {
      // Cancel animation frame on error
      if (animationFrameId !== null) {
        cancelAnimationFrame(animationFrameId)
      }
      
      if (error instanceof DOMException && error.name === 'AbortError') {
        streamCompleted = true
        // Let buffer finish rendering
        await new Promise((resolve) => setTimeout(resolve, 100))
        
        updateLastMessage((prev) => ({
          ...prev,
          content: accumulatedContent || displayedContent || prev.content || '',
          isStreaming: false,
          sources: sources.length > 0 ? sources : prev.sources,
          quality_score:
            qualityScore !== null && qualityScore !== undefined
              ? qualityScore
              : prev.quality_score,
          follow_up_questions:
            followUpQuestions.length > 0 ? followUpQuestions : prev.follow_up_questions,
          context_documents: effectiveContextDocs,
          context_document_labels: effectiveContextDocLabels,
        }))
      } else {
        console.error('Error sending message:', error)
        updateLastMessage(() => ({
          role: 'assistant',
          content: 'Sorry, I encountered an error processing your request. Please try again.',
          isStreaming: false,
          context_documents: effectiveContextDocs,
          context_document_labels: effectiveContextDocLabels,
        }))
      }
    } finally {
      // Ensure animation frame is cancelled
      if (animationFrameId !== null) {
        cancelAnimationFrame(animationFrameId)
      }
      abortControllerRef.current = null
      setIsLoading(false)
    }
  }

  const handleFollowUpClick = (question: string) => {
    handleSendMessage(question, [], [])
  }

  // Get user messages for history navigation
  const userMessages = messages
    .filter((msg) => msg.role === 'user')
    .map((msg) => msg.content)

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mb-4">
              <svg
                className="w-8 h-8 text-primary-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                />
              </svg>
            </div>
            <h2 className="text-xl font-semibold text-secondary-700 mb-2">
              Start a Conversation
            </h2>
            <p className="text-secondary-500 max-w-md">
              Upload some documents and start asking questions. I&apos;ll help you find
              relevant information and provide intelligent answers.
            </p>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto space-y-4">
            {messages.map((message, index) => (
              <div key={index}>
                <MessageBubble message={message} />
                {message.role === 'assistant' &&
                  !message.isStreaming &&
                  message.follow_up_questions &&
                  message.follow_up_questions.length > 0 &&
                  index === messages.length - 1 && (
                    <FollowUpQuestions
                      questions={message.follow_up_questions}
                      onQuestionClick={handleFollowUpClick}
                    />
                  )}
              </div>
            ))}
            {(isLoading || isHistoryLoading) && <LoadingIndicator />}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t border-secondary-200 bg-white px-6 py-4">
        <div className="max-w-4xl mx-auto">
          <ChatInput
            onSend={handleSendMessage}
            onStop={handleStopStreaming}
            disabled={isHistoryLoading}
            isStreaming={isLoading}
            userMessages={userMessages}
          />
        </div>
      </div>
    </div>
  )
}
