export interface StagedDocument {
  file_id: string
  filename: string
  file_size: number
  file_path: string
  timestamp: number
}

export interface ProcessProgress {
  file_id: string
  filename: string
  status: 'processing' | 'completed' | 'error'
  chunks_processed: number
  total_chunks: number
  progress_percentage: number
  error?: string
}

export interface StageDocumentResponse {
  file_id: string
  filename: string
  status: string
  error?: string
}
