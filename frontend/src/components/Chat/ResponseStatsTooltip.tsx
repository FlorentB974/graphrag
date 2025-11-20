'use client'

import { useState } from 'react'
import { InformationCircleIcon } from '@heroicons/react/24/outline'
import { motion } from 'framer-motion'
import { MessageStats, SessionAggregateStats } from '@/types'

interface ResponseStatsTooltipProps {
  stats?: MessageStats
  sessionStats?: SessionAggregateStats
}

const formatNumber = (value?: number, digits = 0) => {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return '—'
  }
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  })
}

const formatMs = (value?: number) => {
  if (value === undefined || value === null) return '—'
  if (value >= 1000) {
    return `${(value / 1000).toFixed(2)} s`
  }
  return `${value.toFixed(0)} ms`
}

const formatCost = (value?: number) => {
  if (value === undefined || value === null) return '—'
  if (value < 0.001) {
    return `$${value.toFixed(4)}`
  }
  return `$${value.toFixed(3)}`
}

export default function ResponseStatsTooltip({ stats, sessionStats }: ResponseStatsTooltipProps) {
  const [isOpen, setIsOpen] = useState(false)

  if (!stats) {
    return null
  }

  const llm = stats.llm
  const pipeline = stats.pipeline
  const retrieval = stats.retrieval
  const stageTimings = pipeline?.stage_timings?.slice(0, 4) || []

  return (
    <div
      className="relative inline-flex items-center"
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
    >
      <button
        type="button"
        className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-secondary-100/90 text-secondary-500 shadow-sm transition-colors hover:bg-white hover:text-secondary-700 dark:bg-secondary-700/80 dark:text-secondary-100"
        onFocus={() => setIsOpen(true)}
        onBlur={() => setIsOpen(false)}
        onClick={() => setIsOpen((prev) => !prev)}
        aria-label="Show response stats"
      >
        <InformationCircleIcon className="h-4 w-4" />
      </button>

      {isOpen && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.2, ease: 'easeOut' }}
          className="absolute right-0 bottom-full z-[100] mb-2 w-80 rounded-xl border border-secondary-200/80 bg-white p-4 text-xs shadow-2xl shadow-black/10 backdrop-blur-sm dark:border-secondary-500 dark:bg-secondary-900"
        >
          <div className="mb-3">
            <p className="text-[0.65rem] uppercase tracking-wide text-secondary-600 dark:text-secondary-300">
              Response
            </p>
            <div className="mt-1 grid grid-cols-2 gap-x-3 gap-y-1">
              <span className="text-secondary-500 dark:text-secondary-400">Tokens</span>
              <span className="text-right font-semibold text-secondary-900 dark:text-white">
                {formatNumber(llm?.total_tokens)}
              </span>
              <span className="text-secondary-500 dark:text-secondary-400">Latency</span>
              <span className="text-right font-semibold text-secondary-900 dark:text-white">
                {formatMs(llm?.latency_ms)}
              </span>
              <span className="text-secondary-500 dark:text-secondary-400">Cost</span>
              <span className="text-right font-semibold text-secondary-900 dark:text-white">
                {formatCost(llm?.total_cost_usd)}
              </span>
            </div>
          </div>

          <div className="mb-3">
            <p className="text-[0.65rem] uppercase tracking-wide text-secondary-600 dark:text-secondary-300">
              Pipeline
            </p>
            <div className="mt-1 flex items-center justify-between text-secondary-900 dark:text-white">
              <span>Total</span>
              <span className="font-semibold">{formatMs(pipeline?.total_duration_ms)}</span>
            </div>
            {stageTimings.length > 0 && (
              <ul className="mt-1 space-y-1">
                {stageTimings.map((stage, index) => (
                  <li
                    key={`${stage.stage}-${stage.timestamp ?? index}`}
                    className="flex items-center justify-between text-secondary-500 dark:text-secondary-300"
                  >
                    <span>{stage.stage.replace(/_/g, ' ')}</span>
                    <span className="text-secondary-400 dark:text-secondary-500">{formatMs(stage.duration_ms)}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="mb-3">
            <p className="text-[0.65rem] uppercase tracking-wide text-secondary-600 dark:text-secondary-300">
              Retrieval
            </p>
            <div className="mt-1 grid grid-cols-2 gap-x-3 gap-y-1">
              <span className="text-secondary-500 dark:text-secondary-400">Mode</span>
              <span className="text-right font-semibold text-secondary-900 dark:text-white">
                {retrieval?.mode || '—'}
              </span>
              <span className="text-secondary-500 dark:text-secondary-400">Chunks</span>
              <span className="text-right font-semibold text-secondary-900 dark:text-white">
                {retrieval?.chunks_used ?? retrieval?.chunks_retrieved ?? '—'}
              </span>
              <span className="text-secondary-500 dark:text-secondary-400">Documents</span>
              <span className="text-right font-semibold text-secondary-900 dark:text-white">
                {retrieval?.unique_documents ?? '—'}
              </span>
            </div>
          </div>

          {sessionStats && (
            <div>
              <p className="text-[0.65rem] uppercase tracking-wide text-secondary-600 dark:text-secondary-300">
                Session totals
              </p>
              <div className="mt-1 grid grid-cols-2 gap-x-3 gap-y-1">
                <span className="text-secondary-500 dark:text-secondary-400">Responses</span>
                <span className="text-right font-semibold text-secondary-900 dark:text-white">
                  {sessionStats.assistantResponses}
                </span>
                <span className="text-secondary-500 dark:text-secondary-400">Tokens</span>
                <span className="text-right font-semibold text-secondary-900 dark:text-white">
                  {formatNumber(sessionStats.totalTokens)}
                </span>
                <span className="text-secondary-500 dark:text-secondary-400">Cost</span>
                <span className="text-right font-semibold text-secondary-900 dark:text-white">
                  {formatCost(sessionStats.totalCostUsd)}
                </span>
                <span className="text-secondary-500 dark:text-secondary-400">Avg latency</span>
                <span className="text-right font-semibold text-secondary-900 dark:text-white">
                  {formatMs(sessionStats.avgLatencyMs)}
                </span>
              </div>
            </div>
          )}
        </motion.div>
      )}
    </div>
  )
}
