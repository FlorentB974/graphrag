import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type Theme = 'light' | 'dark' | 'auto'

interface ThemeStore {
  theme: Theme
  setTheme: (theme: Theme) => void
  isDark: boolean
  setIsDark: (isDark: boolean) => void
}

export const useThemeStore = create<ThemeStore>()(
  persist(
    (set, get) => ({
      theme: 'auto',
      isDark: false,
      setTheme: (theme: Theme) => {
        set({ theme })
        applyTheme(theme, get().isDark)
      },
      setIsDark: (isDark: boolean) => {
        set({ isDark })
        applyTheme(get().theme, isDark)
      },
    }),
    {
      name: 'theme-store',
      partialize: (state) => ({ theme: state.theme }),
    }
  )
)

export function applyTheme(theme: Theme, systemDark: boolean) {
  const html = document.documentElement
  
  let shouldBeDark: boolean
  
  if (theme === 'auto') {
    shouldBeDark = systemDark
  } else {
    shouldBeDark = theme === 'dark'
  }
  
  if (shouldBeDark) {
    html.classList.add('dark')
  } else {
    html.classList.remove('dark')
  }
}

export function initializeTheme() {
  if (typeof window === 'undefined') return
  
  const { setIsDark } = useThemeStore.getState()
  
  // Check system preference
  const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches
  setIsDark(systemDark)
  
  // Listen for system preference changes
  const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
  const handleChange = (e: MediaQueryListEvent) => {
    setIsDark(e.matches)
  }
  
  mediaQuery.addEventListener('change', handleChange)
  return () => mediaQuery.removeEventListener('change', handleChange)
}
