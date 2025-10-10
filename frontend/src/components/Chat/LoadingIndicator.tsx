'use client'

export default function LoadingIndicator() {
  return (
    <div className="flex items-center space-x-2 text-secondary-500">
      <div className="flex space-x-1">
        <div className="w-2 h-2 bg-primary-500 rounded-full animate-bounce"></div>
        <div
          className="w-2 h-2 bg-primary-500 rounded-full animate-bounce"
          style={{ animationDelay: '0.1s' }}
        ></div>
        <div
          className="w-2 h-2 bg-primary-500 rounded-full animate-bounce"
          style={{ animationDelay: '0.2s' }}
        ></div>
      </div>
      <span className="text-sm">Thinking...</span>
    </div>
  )
}
