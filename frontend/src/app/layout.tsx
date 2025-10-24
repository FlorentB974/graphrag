import type { Metadata } from 'next'
import './globals.css'
import ToastContainer from '@/components/Toast/ToastContainer'

export const metadata: Metadata = {
  title: 'GraphRAG - Chat with Your Documents',
  description: 'Intelligent document Q&A powered by graph-based RAG',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>
        {children}
        <ToastContainer />
      </body>
    </html>
  )
}
