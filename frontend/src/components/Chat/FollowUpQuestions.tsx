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
      <div className="flex items-center text-sm font-medium text-secondary-700 dark:text-secondary-300 mb-2">
        <SparklesIcon className="w-4 h-4 mr-1" />
        Follow-up questions
      </div>
      <div className="space-y-2">
        {questions.map((question, index) => (
          <button
            key={index}
            onClick={() => onQuestionClick(question)}
            className="block w-full text-left px-4 py-3 bg-white dark:bg-secondary-800 dark:bg-secondary-800 border border-secondary-300 dark:border-secondary-600 rounded-lg hover:border-primary-500 dark:hover:border-primary-400 hover:bg-primary-50 dark:hover:bg-primary-900/20 transition-all duration-200 text-sm text-secondary-700 dark:text-secondary-300 hover:text-primary-700 dark:hover:text-primary-300 suggestion-shimmer"
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  )
}
