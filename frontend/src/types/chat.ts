export type Confidence = 'high' | 'medium' | 'low' | 'none'

export interface ReferenceItem {
  chunk_id: number
  filename: string
  page: number
  quote: string
  source_excerpt?: string | null
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  confidence?: Confidence
  references?: ReferenceItem[]
  standaloneQuery?: string
  isStreaming?: boolean
  createdAt?: string
}

export interface ChatRequest {
  message: string
  session_id?: number
}

export interface ChatFinalEvent {
  session_id: number
  standalone_query: string
  answer: string
  references: ReferenceItem[]
  confidence: Confidence
  retrieval_score?: number | null
  retrieval_attempts?: number
}

export interface UploadResponse {
  filename: string
  document_id: number
  year: string | null
  quarter: string | null
  pages: number
  chunks: number
  token_estimate?: number
  saved_path: string
}

export interface SessionSummary {
  session_id: number
  title: string
  last_message_preview: string
  last_message_at: string | null
  message_count: number
}

export interface SessionMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  references: ReferenceItem[]
  confidence?: Confidence
  standalone_query?: string
  created_at: string
}

export interface SessionMessagesResponse {
  session_id: number
  messages: SessionMessage[]
}

export interface ChatSearchResult {
  session_id: number
  message_id: number
  role: 'user' | 'assistant'
  snippet: string
  created_at: string
}
