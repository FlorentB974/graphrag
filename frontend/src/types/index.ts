export interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp?: string
  sources?: Source[]
  quality_score?: QualityScore
  follow_up_questions?: string[]
  isStreaming?: boolean
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
}

export interface DocumentSummary {
  document_id: string
  filename: string
  created_at: string
  chunk_count: number
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

export interface ChatRequest {
  message: string
  session_id?: string
  retrieval_mode?: 'hybrid' | 'simple' | 'graph_enhanced'
  top_k?: number
  temperature?: number
  use_multi_hop?: boolean
  stream?: boolean
}
