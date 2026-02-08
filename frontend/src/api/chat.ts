import { fetchEventSource } from '@microsoft/fetch-event-source'

import type {
  ChatFinalEvent,
  ChatRequest,
  ChatSearchResult,
  SessionMessagesResponse,
  SessionSummary,
  UploadResponse,
} from '../types/chat'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

interface StreamHandlers {
  signal: AbortSignal
  onSession: (sessionId: number) => void
  onMeta: (standaloneQuery: string) => void
  onStatus: (stage: string) => void
  onRetrieval: (score: number | null, attempts: number) => void
  onDelta: (delta: string) => void
  onFinal: (payload: ChatFinalEvent) => void
  onError: (message: string) => void
}

function parseJson<T>(raw: string): T | null {
  try {
    return JSON.parse(raw) as T
  } catch {
    return null
  }
}

export async function streamChat(request: ChatRequest, handlers: StreamHandlers): Promise<void> {
  await fetchEventSource(`${API_BASE_URL}/api/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify(request),
    signal: handlers.signal,
    openWhenHidden: true,
    async onopen(response) {
      if (!response.ok) {
        const message = `Streaming failed (${response.status})`
        handlers.onError(message)
        throw new Error(message)
      }
    },
    onmessage(msg) {
      if (!msg.data) {
        return
      }

      if (msg.event === 'session') {
        const payload = parseJson<{ session_id: number }>(msg.data)
        if (payload?.session_id) {
          handlers.onSession(payload.session_id)
        }
        return
      }

      if (msg.event === 'meta') {
        const payload = parseJson<{ standalone_query: string }>(msg.data)
        if (payload?.standalone_query) {
          handlers.onMeta(payload.standalone_query)
        }
        return
      }

      if (msg.event === 'status') {
        const payload = parseJson<{ stage: string }>(msg.data)
        if (payload?.stage) {
          handlers.onStatus(payload.stage)
        }
        return
      }

      if (msg.event === 'retrieval') {
        const payload = parseJson<{ best_score?: number | null; attempts?: { attempt: number }[] }>(msg.data)
        handlers.onRetrieval(payload?.best_score ?? null, payload?.attempts?.length ?? 1)
        return
      }

      if (msg.event === 'delta') {
        const payload = parseJson<{ text: string }>(msg.data)
        if (payload?.text) {
          handlers.onDelta(payload.text)
        }
        return
      }

      if (msg.event === 'final') {
        const payload = parseJson<ChatFinalEvent>(msg.data)
        if (payload) {
          handlers.onFinal(payload)
        }
        return
      }

      if (msg.event === 'error') {
        const payload = parseJson<{ detail?: string }>(msg.data)
        handlers.onError(payload?.detail ?? 'Server error')
      }
    },
    onerror(err) {
      handlers.onError(err instanceof Error ? err.message : 'Unknown streaming error')
      throw err
    },
  })
}

export async function uploadPdf(file: File): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE_URL}/api/upload`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Upload failed (${response.status})`)
  }

  return (await response.json()) as UploadResponse
}

export async function listSessions(): Promise<SessionSummary[]> {
  const response = await fetch(`${API_BASE_URL}/api/sessions`)
  if (!response.ok) {
    throw new Error(`Failed to load sessions (${response.status})`)
  }

  return (await response.json()) as SessionSummary[]
}

export async function fetchSessionMessages(sessionId: number): Promise<SessionMessagesResponse> {
  const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/messages`)
  if (!response.ok) {
    throw new Error(`Failed to load session messages (${response.status})`)
  }

  return (await response.json()) as SessionMessagesResponse
}

export async function deleteAllSessions(): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/sessions`, { method: 'DELETE' })
  if (!response.ok) {
    throw new Error(`Failed to delete sessions (${response.status})`)
  }
}

export async function searchChatMessages(keyword: string): Promise<ChatSearchResult[]> {
  const params = new URLSearchParams({ keyword })
  const response = await fetch(`${API_BASE_URL}/api/sessions/search?${params.toString()}`)
  if (!response.ok) {
    throw new Error(`Failed to search chat messages (${response.status})`)
  }

  return (await response.json()) as ChatSearchResult[]
}
