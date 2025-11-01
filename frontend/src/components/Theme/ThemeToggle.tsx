'use client'

import { useThemeStore, type Theme } from '@/store/themeStore'
import { useEffect, useState } from 'react'
import clsx from 'clsx'

const themes: Theme[] = ['light', 'dark', 'auto']
const themeLabels: Record<Theme, string> = {
  light: 'Light',
  dark: 'Dark',
  auto: 'Auto',
}

export function ThemeToggle() {
  const { theme, setTheme, isDark } = useThemeStore()
  const [mounted, setMounted] = useState(false)
  const [showTooltip, setShowTooltip] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) return null

  const currentIndex = themes.indexOf(theme)
  const nextIndex = (currentIndex + 1) % themes.length
  const nextTheme = themes[nextIndex]

  const isCurrentlyDark = theme === 'dark' || (theme === 'auto' && isDark)

  return (
    <div className="fixed bottom-6 right-6 z-50">
      {/* Tooltip */}
      {showTooltip && (
        <div className="absolute bottom-12 right-0 mb-2 whitespace-nowrap">
          <div className="px-3 py-2 bg-secondary-900 dark:bg-secondary-50 text-white dark:text-secondary-900 text-sm rounded-lg shadow-lg">
            {themeLabels[theme]}
          </div>
        </div>
      )}

      {/* Toggle Button */}
      <button
        onClick={() => setTheme(nextTheme)}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        className={clsx(
          'flex items-center justify-center w-12 h-12 rounded-full',
          'bg-secondary-200 dark:bg-secondary-700',
          'text-secondary-700 dark:text-secondary-200',
          'hover:bg-secondary-300 dark:hover:bg-secondary-600',
          'shadow-lg hover:shadow-xl',
          'transition-all duration-200',
          'border border-secondary-300 dark:border-secondary-600'
        )}
        title={`Switch to ${themeLabels[nextTheme]} mode`}
        aria-label="Toggle theme"
      >
        {!isCurrentlyDark ? (
          // Light icon
          <svg
            className="w-5 h-5"
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path
              fillRule="evenodd"
              d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l-2.828-2.829a1 1 0 00-1.414 1.414l2.828 2.829a1 1 0 001.414-1.414zM2.05 6.464l2.828 2.829a1 1 0 001.414-1.414L3.464 5.05A1 1 0 102.05 6.464zm9.9-9.9L6.464 3.464a1 1 0 001.414 1.414L13.464 2.05a1 1 0 00-1.414-1.414zm0 18.8L6.464 15.536a1 1 0 001.414 1.414l7-7a1 1 0 00-1.414-1.414zM17.95 13.536l-2.828-2.829a1 1 0 10-1.414 1.414l2.828 2.829a1 1 0 001.414-1.414zM16.5 9a1 1 0 011 1v2a1 1 0 11-2 0v-2a1 1 0 011-1zm2.95 2.95a1 1 0 11-1.414-1.414l2-2a1 1 0 111.414 1.414l-2 2z"
              clipRule="evenodd"
            />
          </svg>
        ) : (
          // Dark icon
          <svg
            className="w-5 h-5"
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
          </svg>
        )}
      </button>
    </div>
  )
}
