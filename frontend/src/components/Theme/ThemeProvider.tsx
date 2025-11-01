'use client'

import { useEffect } from 'react'
import { initializeTheme, applyTheme, useThemeStore } from '@/store/themeStore'

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const theme = useThemeStore((state) => state.theme)
  const isDark = useThemeStore((state) => state.isDark)

  useEffect(() => {
    const cleanup = initializeTheme()
    return cleanup
  }, [])

  useEffect(() => {
    applyTheme(theme, isDark)
  }, [theme, isDark])

  return <>{children}</>
}
