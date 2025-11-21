import { FormEvent, useEffect, useRef, useState } from 'react'
import './App.css'

type Source = {
  id?: string
  title?: string
  text?: string
  url?: string
  score?: number
}

type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  isStreaming?: boolean
  stage?: string | null
  sources?: Source[]
  followUps?: string[]
  qualityScore?: number | null
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const createId = () =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`

function MessageBubble({ message, onFollowUp }: { message: ChatMessage; onFollowUp?: (question: string) => void }) {
  const isUser = message.role === 'user'
  const label = isUser ? 'You' : 'GraphRAG'

  return (
    <div className={`message ${isUser ? 'from-user' : 'from-assistant'}`}>
      <div className="message-header">
        <span className="pill">{label}</span>
        <span className="timestamp">{new Date(message.timestamp).toLocaleTimeString()}</span>
      </div>
      <div className="bubble">
        <p className="message-content">{message.content || (message.isStreaming ? '...' : '')}</p>
        {message.isStreaming && (
          <div className="streaming" aria-label="Assistant is responding">
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
            {message.stage && <span className="stage">{message.stage}</span>}
          </div>
        )}
        {typeof message.qualityScore === 'number' && !message.isStreaming && (
          <div className="meta-row">
            <span className="badge">Quality: {message.qualityScore.toFixed(1)}</span>
          </div>
        )}
        {Array.isArray(message.sources) && message.sources.length > 0 && (
          <div className="sources">
            <div className="sources-title">Sources</div>
            <div className="sources-grid">
              {message.sources.map((source, index) => (
                <div className="source-card" key={`${source.id ?? index}`}>
                  <div className="source-title">{source.title || source.url || 'Referenced document'}</div>
                  {typeof source.score === 'number' && (
                    <div className="source-score">Relevance {Math.round(source.score * 100)}%</div>
                  )}
                  {source.text && <p className="source-text">{source.text}</p>}
                  {source.url && (
                    <a className="source-link" href={source.url} target="_blank" rel="noreferrer">
                      Open
                    </a>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
        {Array.isArray(message.followUps) && message.followUps.length > 0 && !message.isStreaming && (
          <div className="followups">
            <div className="followups-title">Try asking</div>
            <div className="followups-buttons">
              {message.followUps.map((question) => (
                <button
                  key={question}
                  type="button"
                  className="followup-chip"
                  onClick={() => onFollowUp?.(question)}
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [isSending, setIsSending] = useState(false)
  const [currentStage, setCurrentStage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const messagesEndRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_URL}/api/health`)
        setIsHealthy(response.ok)
      } catch (err) {
        console.error('Health check failed', err)
        setIsHealthy(false)
      }
    }

    checkHealth()
    const timer = setInterval(checkHealth, 30000)
    return () => clearInterval(timer)
  }, [])

  const updateAssistantMessage = (id: string, updater: (previous: ChatMessage) => ChatMessage) => {
    setMessages((prev) => prev.map((message) => (message.id === id ? updater(message) : message)))
  }

  const handleSend = async (event?: FormEvent, promptText?: string) => {
    event?.preventDefault()
    const text = (promptText ?? input).trim()
    if (!text || isSending) return

    const userMessage: ChatMessage = {
      id: createId(),
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    }
    const assistantId = createId()
    const placeholder: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
      isStreaming: true,
      stage: 'Analyzing request',
    }

    setMessages((prev) => [...prev, userMessage, placeholder])
    if (!promptText) {
      setInput('')
    }
    setIsSending(true)
    setError(null)
    setCurrentStage('Analyzing request')

    let accumulatedContent = ''
    let sources: Source[] = []
    let followUps: string[] = []
    let qualityScore: number | null = null
    let newSessionId = sessionId
    let stageLabel: string | null = null

    const controller = new AbortController()
    abortControllerRef.current = controller

    try {
      const response = await fetch(`${API_URL}/api/chat/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: text,
          session_id: sessionId || undefined,
          stream: true,
        }),
        signal: controller.signal,
      })

      if (!response.ok || !response.body) {
        throw new Error(`Chat request failed (${response.status})`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n').filter(Boolean)

        for (const line of lines) {
          if (!line.startsWith('data:')) continue
          try {
            const payload = JSON.parse(line.replace(/^data:\s*/, ''))
            switch (payload.type) {
              case 'token': {
                const contentPiece = payload.content ?? ''
                accumulatedContent += contentPiece
                updateAssistantMessage(assistantId, (prev) => ({
                  ...prev,
                  content: accumulatedContent,
                  isStreaming: true,
                  stage: stageLabel ?? prev.stage,
                }))
                break
              }
              case 'sources':
                sources = Array.isArray(payload.content) ? payload.content : []
                break
              case 'follow_ups':
                followUps = Array.isArray(payload.content) ? payload.content : []
                break
              case 'quality_score':
                qualityScore = typeof payload.content === 'number' ? payload.content : qualityScore
                break
              case 'metadata':
                if (payload.content?.session_id) {
                  newSessionId = payload.content.session_id
                }
                break
              case 'stage':
                stageLabel = payload.content || null
                setCurrentStage(stageLabel)
                updateAssistantMessage(assistantId, (prev) => ({
                  ...prev,
                  stage: stageLabel,
                }))
                break
              case 'error':
                throw new Error(payload.content || 'The assistant returned an error')
              case 'done':
                break
              default:
                break
            }
          } catch (err) {
            console.error('Failed to parse SSE line', err)
          }
        }
      }

      updateAssistantMessage(assistantId, (prev) => ({
        ...prev,
        content: accumulatedContent || prev.content,
        isStreaming: false,
        stage: stageLabel,
        sources,
        followUps,
        qualityScore,
      }))

      if (newSessionId && newSessionId !== sessionId) {
        setSessionId(newSessionId)
      }
    } catch (err) {
      console.error(err)
      if (err instanceof DOMException && err.name === 'AbortError') {
        updateAssistantMessage(assistantId, (prev) => ({
          ...prev,
          content: accumulatedContent || prev.content,
          isStreaming: false,
          stage: stageLabel,
          sources,
          followUps,
          qualityScore,
        }))
      } else {
        const message = err instanceof Error ? err.message : 'Unexpected error'
        setError(message)
        updateAssistantMessage(assistantId, (prev) => ({
          ...prev,
          content: 'Sorry, something went wrong. Please try again.',
          isStreaming: false,
        }))
      }
    } finally {
      abortControllerRef.current = null
      setIsSending(false)
      setCurrentStage(null)
    }
  }

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
  }

  const handleNewChat = () => {
    handleStop()
    setMessages([])
    setSessionId(null)
    setCurrentStage(null)
    setError(null)
  }

  const handleFollowUp = (question: string) => {
    setInput('')
    handleSend(undefined, question)
  }

  return (
    <div className="page">
      <header className="topbar">
        <div>
          <p className="eyebrow">GraphRAG assistant</p>
          <h1>User chat experience</h1>
          <p className="lede">
            Ask questions about your connected documents. Streaming responses and source highlights help users stay focused on
            answers.
          </p>
        </div>
        <div className="status">
          <div className={`status-pill ${isHealthy === false ? 'status-bad' : 'status-good'}`}>
            <span className={`dot ${isHealthy === false ? 'dot-bad' : 'dot-good'}`} />
            {isHealthy === false ? 'Offline' : 'Connected'}
          </div>
          {sessionId && <span className="session">Session: {sessionId}</span>}
          <div className="actions">
            <button className="ghost" type="button" onClick={handleNewChat}>
              Start fresh
            </button>
            <button className="secondary" type="button" onClick={handleStop} disabled={!isSending}>
              Stop response
            </button>
          </div>
        </div>
      </header>

      <main className="chat-card">
        <div className="conversation" role="log" aria-live="polite">
          {messages.length === 0 ? (
            <div className="empty">
              <p className="empty-title">Welcome</p>
              <p className="empty-copy">Start chatting to explore answers from your knowledge base.</p>
              <div className="empty-hint">
                <span>Need inspiration?</span>
                <ul>
                  <li>"Summarize the latest updates in the project plan"</li>
                  <li>"What are the key risks mentioned in the docs?"</li>
                  <li>"Give me follow-up questions to explore this topic."</li>
                </ul>
              </div>
            </div>
          ) : (
            messages.map((message) => (
              <MessageBubble key={message.id} message={message} onFollowUp={handleFollowUp} />
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {currentStage && (
          <div className="stage-banner">Current stage: {currentStage}</div>
        )}

        {error && <div className="error">{error}</div>}

        <form className="input-row" onSubmit={handleSend}>
          <input
            aria-label="Message"
            placeholder="Ask anything about your data..."
            value={input}
            onChange={(event) => setInput(event.target.value)}
            disabled={isSending}
          />
          <button className="primary" type="submit" disabled={isSending || !input.trim()}>
            {isSending ? 'Sending...' : 'Send'}
          </button>
        </form>
      </main>
    </div>
  )
}

export default App
