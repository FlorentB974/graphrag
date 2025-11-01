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
              d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z"
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
