export interface StagedDocument {
  file_id: string
  filename: string
  file_size: number
  file_path: string
  timestamp: number
  document_id?: string
  mode?: 'full' | 'chunks_only' | 'entities_only'
}

export interface ProcessProgress {
  file_id: string
  document_id?: string
  filename: string
  status: 'queued' | 'processing' | 'completed' | 'error'
  stage?: string | null
  mode?: 'full' | 'chunks_only' | 'entities_only'
  queue_position?: number | null
  chunks_processed: number
  total_chunks: number
  chunk_progress?: number | null
  entity_progress?: number | null
  progress_percentage: number
  entity_state?: string | null
  error?: string
}

export interface StageDocumentResponse {
  file_id: string
  filename: string
  document_id?: string
  status: string
  error?: string
}

export interface ProcessingGlobalSummary {
  is_processing: boolean
  current_file_id?: string | null
  current_document_id?: string | null
  current_filename?: string | null
  current_stage?: string | null
  progress_percentage?: number | null
  queue_length: number
  pending_documents: ProcessProgress[]
}

export interface ProcessingProgressResponse {
  progress: ProcessProgress[]
  global: ProcessingGlobalSummary
}
