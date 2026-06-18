/** Mirrors the pydantic schemas exposed by the RAGBot backend (Phase 5 API). */

export type DocumentStatus = 'pending' | 'processing' | 'ready' | 'failed'

/** Named `DocumentRecord` to avoid colliding with the DOM's global `Document`. */
export interface DocumentRecord {
  id: string
  filename: string
  uploaded_at: string
  num_chunks: number
  status: DocumentStatus
}

export interface AskRequest {
  question: string
  document_ids?: string[]
  /** Continue an existing conversation; omit to start a new one. */
  conversation_id?: string
}

export interface Source {
  citation: number
  document_id: string
  filename: string
  chunk_index: number
  /** 1-based page the passage starts on (PDF); null for TXT/MD. */
  page: number | null
  char_start: number
  char_end: number
  /** Full retrieved chunk text (not a truncated snippet). */
  text: string
  score: number
}

/** How well an answer is grounded in the indexed documents. */
export type GroundingStatus = 'answered' | 'insufficient_context' | 'no_documents'

export interface AnswerResult {
  answer: string | null
  sources: Source[]
  generation_enabled: boolean
  /** Grounding outcome — drives the honesty UI state for the turn. */
  status: GroundingStatus
  /** Best-source similarity, clamped to 0..1; the confidence indicator. */
  coverage: number
  /** Answer produced but cites no valid source — possible ungrounded guess. */
  low_confidence: boolean
  cited: number[]
  conversation_id: string
  /** The follow-up rewritten as a standalone question for retrieval, when
   * prior turns existed; null on the first turn or when generation is off. */
  standalone_question: string | null
}

export interface ProviderStatus {
  provider: string
  embeddings_enabled: boolean
  generation_enabled: boolean
  embed_model: string | null
  chat_model: string | null
  reachable: boolean
}

export interface UploadLimits {
  max_upload_size_bytes: number
  max_files_per_request: number
  allowed_extensions: string[]
}

export interface HealthResponse {
  status: string
  version: string
  providers: ProviderStatus
  limits: UploadLimits
}
