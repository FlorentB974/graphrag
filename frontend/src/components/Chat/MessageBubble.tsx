'use client'

import { Message } from '@/types'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'
import SourcesList from './SourcesList'
import QualityBadge from './QualityBadge'
import { motion } from 'framer-motion'

interface MessageBubbleProps {
  message: Message
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user'

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{
        duration: 0.4,
        ease: [0.4, 0, 0.2, 1],
      }}
      className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      <div
        className={`chat-message ${
          isUser ? 'chat-message-user' : 'chat-message-assistant relative pr-12'
        }`}
      >

        <div className={isUser ? '' : 'prose prose-sm prose-slate max-w-none'}>
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <ReactMarkdown 
              remarkPlugins={[remarkGfm, remarkBreaks]}
            >
              {message.content}
            </ReactMarkdown>
          )}
        </div>

        {message.isStreaming && (
          <div className="flex items-center mt-2 text-secondary-500">
            <div className="flex space-x-1">
              <div className="w-2 h-2 bg-secondary-400 rounded-full animate-bounce"></div>
              <div
                className="w-2 h-2 bg-secondary-400 rounded-full animate-bounce"
                style={{ animationDelay: '0.1s' }}
              ></div>
              <div
                className="w-2 h-2 bg-secondary-400 rounded-full animate-bounce"
                style={{ animationDelay: '0.2s' }}
              ></div>
            </div>
          </div>
        )}

        {/* bottom area: sources list (left) and quality badge (anchored) */}
        {!isUser && ( (message.sources && message.sources.length > 0) || message.quality_score ) && (
          <div className="mt-4 pt-4 border-t border-secondary-200 relative">
            {message.sources && message.sources.length > 0 ? (
              <div className="min-w-0">
                <SourcesList sources={message.sources} />
              </div>
            ) : null}

            {message.quality_score && (
              // If sources are present, anchor badge to the top-right of the sources block
              // so it stays on the same horizontal line as the Sources header when expanded.
              // Otherwise, position it at the bubble's bottom-right.
              <div
                className={
                  message.sources && message.sources.length > 0
                    ? 'absolute top-4 right-3 pointer-events-auto'
                    : 'absolute bottom-3 right-3 pointer-events-auto'
                }
              >
                <QualityBadge score={message.quality_score} />
              </div>
            )}
          </div>
        )}
      </div>
    </motion.div>
  )
}
