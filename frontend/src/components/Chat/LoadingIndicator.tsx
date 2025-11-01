'use client'

import { useEffect, useState } from 'react'

interface Stage {
  id: string
  label: string
  emoji: string
  color: string
}

const STAGES: Record<string, Stage> = {
  query_analysis: {
    id: 'query_analysis',
    label: 'Analyzing Query',
    emoji: '🔍',
    color: 'from-blue-500 to-cyan-500',
  },
  retrieval: {
    id: 'retrieval',
    label: 'Retrieving Documents',
    emoji: '📚',
    color: 'from-purple-500 to-pink-500',
  },
  graph_reasoning: {
    id: 'graph_reasoning',
    label: 'Graph Reasoning',
    emoji: '🧠',
    color: 'from-indigo-500 to-purple-500',
  },
  generation: {
    id: 'generation',
    label: 'Generating Response',
    emoji: '✍️',
    color: 'from-yellow-500 to-orange-500',
  },
  quality_calculation: {
    id: 'quality_calculation',
    label: 'Quality Check',
    emoji: '✅',
    color: 'from-green-400 to-green-600',
  },
  suggestions: {
    id: 'suggestions',
    label: 'Preparing Suggestions',
    emoji: '💡',
    color: 'from-pink-500 to-rose-500',
  },
}

interface LoadingIndicatorProps {
  currentStage?: string
  completedStages?: string[]
  isLoading?: boolean
}

export default function LoadingIndicator({ 
  currentStage = 'query_analysis',
  completedStages = [],
  isLoading = true
}: LoadingIndicatorProps) {
  const [displayedStage, setDisplayedStage] = useState<string>(currentStage)
  const [stageHistory, setStageHistory] = useState<string[]>(completedStages)

  useEffect(() => {
    // Update when currentStage changes (during loading)
    if (isLoading && currentStage) {
      setDisplayedStage(currentStage)
      setStageHistory((prev) => {
        if (!prev.includes(currentStage)) {
          return [...prev, currentStage]
        }
        return prev
      })
    }
  }, [currentStage, isLoading])

  // Update stage history when completed stages change (after loading)
  useEffect(() => {
    if (!isLoading && completedStages.length > 0) {
      setStageHistory(completedStages)
      // Set displayed stage to the last completed stage
      setDisplayedStage(completedStages[completedStages.length - 1])
    }
  }, [completedStages, isLoading])

  const stage = STAGES[displayedStage] || STAGES.query_analysis
  const completedStagesCount = isLoading ? stageHistory.slice(0, -1).length : stageHistory.length
  
  const completedStagesPercent = (stageHistory.length / Object.keys(STAGES).length) * 100
  
  // If not loading, show minimal clean display
  if (!isLoading) {
    return (
      <div className="w-full space-y-2">
        {/* Progress bar at 100% - solid green */}
        <div className="relative h-1 bg-secondary-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-green-400 to-green-600"
            style={{ width: '100%' }}
          ></div>
        </div>

        {/* Stage history dots - all completed are green, skipped are grey */}
        <div className="flex gap-2 items-center justify-center py-2">
          {Object.values(STAGES).map((s) => {
            const isCompleted = stageHistory.includes(s.id)
            return (
              <div
                key={s.id}
                className="cursor-help group relative"
                title={s.label}
              >
                <div
                  className={`w-2 h-2 rounded-full transition-all duration-300 ${
                    isCompleted ? 'bg-green-500' : 'bg-secondary-300'
                  }`}
                ></div>
                <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 px-2 py-1 bg-secondary-800 text-white text-xs rounded whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                  {s.label}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    )
  }

  // Loading state - show full interactive display with animations
  return (
    <div className="w-full">
      <style>{`
        @keyframes pulse-glow {
          0%, 100% {
            box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7);
          }
          50% {
            box-shadow: 0 0 0 10px rgba(59, 130, 246, 0);
          }
        }

        @keyframes slide-in {
          from {
            opacity: 0;
            transform: translateX(-10px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }

        @keyframes bounce-smooth {
          0%, 100% {
            transform: translateY(0);
          }
          50% {
            transform: translateY(-4px);
          }
        }

        @keyframes shimmer {
          0% {
            background-position: -1000px 0;
          }
          100% {
            background-position: 1000px 0;
          }
        }

        .animate-pulse-glow {
          animation: pulse-glow 2s infinite;
        }

        .animate-slide-in {
          animation: slide-in 0.3s ease-out;
        }

        .animate-bounce-smooth {
          animation: bounce-smooth 1.5s ease-in-out infinite;
        }

        .animate-shimmer {
          background: linear-gradient(
            90deg,
            rgba(255, 255, 255, 0) 0%,
            rgba(255, 255, 255, 0.3) 50%,
            rgba(255, 255, 255, 0) 100%
          );
          background-size: 1000px 100%;
          animation: shimmer 2s infinite;
        }
      `}</style>

      <div className="space-y-3">
        {/* Current stage with pulsing indicator */}
        <div className="flex items-center gap-3 animate-slide-in">
          <div className="relative">
            <div className="w-3 h-3 bg-primary-500 rounded-full animate-pulse-glow"></div>
            <div className="absolute inset-0 w-3 h-3 bg-primary-500 rounded-full opacity-30"></div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-lg animate-bounce-smooth">{stage.emoji}</span>
            <span className="text-sm font-medium text-secondary-700 dark:text-secondary-300">{stage.label}</span>
          </div>
        </div>

        {/* Progress bar with gradient */}
        <div className="relative h-1 bg-secondary-200 dark:bg-secondary-700 rounded-full overflow-hidden">
          <div
            className={`h-full bg-gradient-to-r ${stage.color} animate-shimmer transition-all duration-500 ease-out`}
            style={{
              width: `${((completedStagesCount + 1) / Object.keys(STAGES).length) * 100}%`,
            }}
          ></div>
        </div>

        {/* Stage history dots */}
        <div className="flex gap-2 items-center justify-center py-2">
          {Object.values(STAGES).map((s) => {
            const isCompleted = stageHistory.includes(s.id)
            const isCurrent = s.id === displayedStage

            return (
              <div
                key={s.id}
                className="cursor-help group relative transition-all duration-300"
                title={s.label}
              >
                <div
                  className={`w-2 h-2 rounded-full transition-all duration-300 ${
                    isCompleted
                      ? 'bg-green-500 scale-100'
                      : isCurrent
                        ? 'bg-primary-500 dark:bg-primary-400 scale-125 animate-pulse-glow'
                        : 'bg-secondary-300 dark:bg-secondary-600 scale-75'
                  }`}
                ></div>
                <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 px-2 py-1 bg-secondary-800 text-white text-xs rounded whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                  {s.label}
                </div>
              </div>
            )
          })}
        </div>

            {/* Processing indicator with animated dots */}
        <div className="flex items-center justify-center gap-1 text-xs text-secondary-500 dark:text-secondary-400 dark:text-secondary-400">
          <span>Processing</span>
          <span className="inline-flex gap-0.5">
            <span className="animate-bounce" style={{ animationDelay: '0s' }}>
              •
            </span>
            <span className="animate-bounce" style={{ animationDelay: '0.15s' }}>
              •
            </span>
            <span className="animate-bounce" style={{ animationDelay: '0.3s' }}>
              •
            </span>
          </span>
        </div>
      </div>
    </div>
  )
}
