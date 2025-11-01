'use client'

import { QualityScore } from '@/types'

interface QualityBadgeProps {
  score: QualityScore
}

export default function QualityBadge({ score }: QualityBadgeProps) {
  const getColor = (value: number) => {
    if (value >= 80) return 'text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/30'
    if (value >= 60) return 'text-yellow-600 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-900/30'
    return 'text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/30'
  }

  const getEmoji = (value: number) => {
    if (value >= 80) return '🟢'
    if (value >= 60) return '🟡'
    return '🔴'
  }

  return (
    <div
      className="inline-flex items-center space-x-2 text-xs"
      style={{ transform: 'translateY(-5px)' }}
    >
      <span>{getEmoji(score.total)}</span>
      <span className={`px-2 py-1 rounded-full font-medium ${getColor(score.total)}`}>
        Quality: {score.total.toFixed(0)}%
      </span>
    </div>
  )
}
