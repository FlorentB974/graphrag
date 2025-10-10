import type { Metadata } from 'next'
import './globals.css'

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
      <body>{children}</body>
    </html>
  )
}
