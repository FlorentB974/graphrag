'use client'

import { QualityScore } from '@/types'

interface QualityBadgeProps {
  score: QualityScore
}

export default function QualityBadge({ score }: QualityBadgeProps) {
  const getColor = (value: number) => {
    if (value >= 80) return 'text-green-600 bg-green-50'
    if (value >= 60) return 'text-yellow-600 bg-yellow-50'
    return 'text-red-600 bg-red-50'
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
