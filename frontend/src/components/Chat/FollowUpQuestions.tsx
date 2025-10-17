'use client'

import { SparklesIcon } from '@heroicons/react/24/outline'

interface FollowUpQuestionsProps {
  questions: string[]
  onQuestionClick: (question: string) => void
}

export default function FollowUpQuestions({
  questions,
  onQuestionClick,
}: FollowUpQuestionsProps) {
  return (
    <div className="mt-4 ml-0 animate-slide-up">
      <div className="flex items-center text-sm font-medium text-secondary-700 mb-2">
        <SparklesIcon className="w-4 h-4 mr-1" />
        Follow-up questions
      </div>
      <div className="space-y-2">
        {questions.map((question, index) => (
          <button
            key={index}
            onClick={() => onQuestionClick(question)}
            className="block w-full text-left px-4 py-3 bg-white border border-secondary-300 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-all duration-200 text-sm text-secondary-700 hover:text-primary-700 suggestion-shimmer"
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  )
}
