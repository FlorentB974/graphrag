import type { ProcessProgress } from './upload'

export interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp?: string
  sources?: Source[]
  quality_score?: QualityScore
  follow_up_questions?: string[]
  isStreaming?: boolean
  context_documents?: string[]
  context_document_labels?: string[]
  context_hashtags?: string[]
}

export interface Source {
  chunk_id?: string
  entity_id?: string
  entity_name?: string
  content: string
  similarity: number
  relevance_score?: number
  document_name: string
  document_id?: string
  filename: string
  chunk_index?: number
  contained_entities?: string[]
  metadata?: Record<string, any>
}

export interface QualityScore {
  total: number
  breakdown: {
    context_relevance: number
    answer_completeness: number
    factual_grounding: number
    coherence: number
    citation_quality: number
  }
  confidence: 'low' | 'medium' | 'high'
}

export interface ChatSession {
  session_id: string
  created_at: string
  updated_at: string
  message_count: number
  preview?: string
}

export interface DatabaseStats {
  total_documents: number
  total_chunks: number
  total_entities: number
  total_relationships: number
  documents: DocumentSummary[]
  processing?: ProcessingSummary
}

export interface DocumentSummary {
  document_id: string
  filename: string
  created_at: string
  chunk_count: number
  processing_status?: string
  processing_stage?: string
  processing_progress?: number
  queue_position?: number | null
  hashtags?: string[]
}

export interface ProcessingSummary {
  is_processing: boolean
  current_file_id?: string | null
  current_document_id?: string | null
  current_filename?: string | null
  current_stage?: string | null
  progress_percentage?: number | null
  queue_length: number
  pending_documents: ProcessProgress[]
}

export interface UploaderInfo {
  id?: string
  name?: string
}

export interface DocumentChunk {
  id: string | number
  text: string
  index?: number
  offset?: number
  score?: number | null
}

export interface DocumentEntity {
  type: string
  text: string
  count?: number
  positions?: Array<number>
}

export interface RelatedDocument {
  id: string
  title?: string
  link?: string
}

export interface DocumentDetails {
  id: string
  title?: string
  file_name?: string
  mime_type?: string
  preview_url?: string
  uploaded_at?: string
  uploader?: UploaderInfo | null
  summary?: string | null
  document_type?: string | null
  hashtags?: string[]
  chunks: DocumentChunk[]
  entities: DocumentEntity[]
  quality_scores?: Record<string, any> | null
  related_documents?: RelatedDocument[]
  metadata?: Record<string, any>
}

export interface UploadResponse {
  filename: string
  status: string
  chunks_created: number
  document_id?: string
  error?: string
}

export interface DocumentTextPayload {
  document_id: string
  text: string
}

export interface ChatRequest {
  message: string
  session_id?: string
  retrieval_mode?: 'hybrid' | 'simple' | 'graph_enhanced'
  top_k?: number
  temperature?: number
  use_multi_hop?: boolean
  stream?: boolean
  context_documents?: string[]
}
