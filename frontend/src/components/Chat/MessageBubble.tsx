'use client'

import { Message, SessionAggregateStats } from '@/types'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'
import SourcesList from './SourcesList'
import QualityBadge from './QualityBadge'
import { motion } from 'framer-motion'
import ResponseStatsTooltip from './ResponseStatsTooltip'

interface MessageBubbleProps {
  message: Message
  sessionStats?: SessionAggregateStats
}

export default function MessageBubble({ message, sessionStats }: MessageBubbleProps) {
  const isUser = message.role === 'user'
  const contextDocDisplay =
    message.context_document_labels && message.context_document_labels.length > 0
      ? message.context_document_labels
      : message.context_documents

  const hasSources = Boolean(message.sources && message.sources.length > 0)
  const hasQualityScore = Boolean(message.quality_score)
  const hasStats = Boolean(message.stats)

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.4, ease: [0.4, 0, 0.2, 1] }}
      className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      <div
        className={`chat-message ${
          isUser ? 'chat-message-user' : 'chat-message-assistant relative pr-4 pl-4'
        }`}
      >
        <div
          className={
            isUser
              ? ''
              : 'prose prose-sm prose-slate dark:prose-invert max-w-none dark:text-secondary-100'
          }
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
              {message.content}
            </ReactMarkdown>
          )}
        </div>

        {isUser &&
          Array.isArray(message.context_documents) &&
          message.context_documents.length > 0 &&
          contextDocDisplay &&
          contextDocDisplay.length > 0 && (
            <div className="mt-2">
              <span className="inline-flex flex-wrap items-center gap-1 rounded-lg bg-white dark:bg-secondary-800/15 px-3 py-1 text-xs text-white/90 max-w-full">
                <span className="font-semibold uppercase tracking-wide text-white/70 shrink-0">
                  {message.context_hashtags && message.context_hashtags.length > 0
                    ? `${message.context_hashtags
                        .map((tag) => (tag.startsWith('#') ? tag : `#${tag}`))
                        .join(', ')}:`
                    : contextDocDisplay.length > 1
                      ? 'Documents:'
                      : 'Document:'}
                </span>
                <span className="break-words" title={contextDocDisplay.join(', ')}>
                  {contextDocDisplay.join(', ')}
                </span>
              </span>
            </div>
          )}

        {message.isStreaming && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
            className="flex items-center mt-2 text-secondary-500 dark:text-secondary-400"
          >
            <motion.div
              animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
              className="w-3 h-3 bg-secondary-400 rounded-full"
            />
          </motion.div>
        )}

        {!isUser && (hasSources || hasQualityScore || hasStats) && (
          <div className="mt-4 pt-4 border-t border-secondary-200 dark:border-secondary-700 relative">
            {hasSources && message.sources && (
              <div>
                <SourcesList sources={message.sources} />
              </div>
            )}

            {(hasQualityScore || hasStats) && (
              <div className="absolute top-[12px] right-0 flex items-center gap-2">
                {hasQualityScore && message.quality_score && (
                  <QualityBadge score={message.quality_score} />
                )}
                <ResponseStatsTooltip stats={message.stats} sessionStats={sessionStats} />
              </div>
            )}
          </div>
        )}
      </div>
    </motion.div>
  )
}
