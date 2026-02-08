import { useCallback, useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import {
  deleteAllSessions,
  deleteSelectedSessions,
  fetchSessionMessages,
  listSessions,
  searchChatMessages,
  streamChat,
} from '../api/chat'
import type { ChatMessage, ChatSearchResult, SessionMessage } from '../types/chat'

function createMessageId(): string {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function mapSessionMessage(message: SessionMessage): ChatMessage {
  return {
    id: String(message.id),
    role: message.role,
    content: message.content,
    confidence: message.confidence,
    references: message.references,
    standaloneQuery: message.standalone_query,
    createdAt: message.created_at,
  }
}

export function useChat() {
  const queryClient = useQueryClient()
  const abortRef = useRef<AbortController | null>(null)

  const [activeSessionId, setActiveSessionId] = useState<number | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [standaloneQuery, setStandaloneQuery] = useState<string>('')
  const [error, setError] = useState<string>('')
  const [statusText, setStatusText] = useState<string>('')
  const [retrievalScore, setRetrievalScore] = useState<number | null>(null)
  const [retrievalAttempts, setRetrievalAttempts] = useState<number>(1)
  const [searchResults, setSearchResults] = useState<ChatSearchResult[]>([])

  const appendDelta = useCallback((assistantId: string, delta: string) => {
    if (!delta) {
      return
    }

    setMessages((prev) =>
      prev.map((item) => {
        if (item.id !== assistantId) {
          return item
        }
        return {
          ...item,
          content: `${item.content}${delta}`,
        }
      }),
    )
  }, [])

  const sessionsQuery = useQuery({
    queryKey: ['sessions'],
    queryFn: listSessions,
    staleTime: 5_000,
  })

  const sessionMessagesQuery = useQuery({
    queryKey: ['sessionMessages', activeSessionId],
    queryFn: () => fetchSessionMessages(activeSessionId as number),
    enabled: activeSessionId !== null,
    staleTime: 2_000,
  })

  useEffect(() => {
    if (activeSessionId === null) {
      return
    }

    if (sessionMessagesQuery.data && !abortRef.current) {
      setMessages(sessionMessagesQuery.data.messages.map(mapSessionMessage))
    }
  }, [activeSessionId, sessionMessagesQuery.data])

  const sendMutation = useMutation({
    mutationFn: async (rawMessage: string) => {
      const message = rawMessage.trim()
      if (!message) {
        return
      }

      setError('')
      setStandaloneQuery('')
      setStatusText('생각 중...')
      setRetrievalScore(null)
      setRetrievalAttempts(1)

      const controller = new AbortController()
      abortRef.current = controller

      let resolvedSessionId = activeSessionId
      const userMessageId = createMessageId()
      const assistantMessageId = createMessageId()

      setMessages((prev) => [
        ...prev,
        { id: userMessageId, role: 'user', content: message },
        { id: assistantMessageId, role: 'assistant', content: '', isStreaming: true },
      ])

      const updateAssistant = (updater: (current: ChatMessage) => ChatMessage) => {
        setMessages((prev) =>
          prev.map((item) => {
            if (item.id !== assistantMessageId) {
              return item
            }
            return updater(item)
          }),
        )
      }

      try {
        await streamChat(
          {
            message,
            session_id: resolvedSessionId ?? undefined,
          },
          {
            signal: controller.signal,
            onSession: (incomingSessionId) => {
              resolvedSessionId = incomingSessionId
              setActiveSessionId(incomingSessionId)
            },
            onMeta: (query) => {
              setStandaloneQuery(query)
            },
            onStatus: (stage) => {
              setStatusText(stage === 'retrieval' ? '검색 중...' : '생각 중...')
            },
            onRetrieval: (score, attempts) => {
              setRetrievalScore(score)
              setRetrievalAttempts(attempts)
            },
            onDelta: (delta) => {
              appendDelta(assistantMessageId, delta)
            },
            onFinal: (payload) => {
              setRetrievalScore(payload.retrieval_score ?? null)
              setRetrievalAttempts(payload.retrieval_attempts ?? 1)
              updateAssistant((current) => ({
                ...current,
                content: current.content.length > 0 ? current.content : payload.answer,
                references: payload.references,
                confidence: payload.confidence,
                standaloneQuery: payload.standalone_query,
                isStreaming: false,
              }))
              setStatusText('')
            },
            onError: (messageText) => {
              setError(messageText)
              updateAssistant((current) => ({
                ...current,
                content: `오류가 발생했습니다: ${messageText}`,
                isStreaming: false,
              }))
              setStatusText('')
            },
          },
        )
      } catch (streamError) {
        if (!controller.signal.aborted) {
          const messageText = streamError instanceof Error ? streamError.message : 'Unknown streaming error'
          setError(messageText)
          updateAssistant((current) => ({
            ...current,
            content: `오류가 발생했습니다: ${messageText}`,
            isStreaming: false,
          }))
        } else {
          updateAssistant((current) => ({
            ...current,
            isStreaming: false,
          }))
        }
      } finally {
        setStatusText('')
        abortRef.current = null

        await queryClient.invalidateQueries({ queryKey: ['sessions'] })
        if (resolvedSessionId !== null) {
          await queryClient.invalidateQueries({ queryKey: ['sessionMessages', resolvedSessionId] })
        }
      }
    },
  })

  const searchMutation = useMutation({
    mutationFn: async (keyword: string) => searchChatMessages(keyword),
    onSuccess: (rows) => {
      setSearchResults(rows)
    },
    onError: (err) => {
      setError(err instanceof Error ? err.message : '검색 중 오류가 발생했습니다.')
    },
  })

  const deleteAllMutation = useMutation({
    mutationFn: deleteAllSessions,
    onSuccess: async () => {
      setActiveSessionId(null)
      setMessages([])
      setSearchResults([])
      setStandaloneQuery('')
      setError('')
      await queryClient.invalidateQueries({ queryKey: ['sessions'] })
    },
    onError: (err) => {
      setError(err instanceof Error ? err.message : '채팅 삭제 중 오류가 발생했습니다.')
    },
  })

  const deleteSelectedMutation = useMutation({
    mutationFn: async (sessionIds: number[]) => deleteSelectedSessions(sessionIds),
    onSuccess: async (_data, sessionIds) => {
      const removed = new Set(sessionIds)
      if (activeSessionId !== null && removed.has(activeSessionId)) {
        setActiveSessionId(null)
        setMessages([])
        setStandaloneQuery('')
        setStatusText('')
        setRetrievalScore(null)
        setRetrievalAttempts(1)
      }
      setSearchResults([])
      setError('')
      await queryClient.invalidateQueries({ queryKey: ['sessions'] })
      if (activeSessionId !== null) {
        await queryClient.invalidateQueries({ queryKey: ['sessionMessages', activeSessionId] })
      }
    },
    onError: (err) => {
      setError(err instanceof Error ? err.message : '채팅 삭제 중 오류가 발생했습니다.')
    },
  })

  const sendMessage = useCallback(
    async (text: string) => {
      await sendMutation.mutateAsync(text)
    },
    [sendMutation],
  )

  const cancelStreaming = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setMessages((prev) =>
      prev.map((item) => (item.isStreaming ? { ...item, isStreaming: false } : item)),
    )
    setStatusText('')
  }, [])

  const selectSession = useCallback(
    async (sessionId: number) => {
      if (abortRef.current) {
        return
      }
      setActiveSessionId(sessionId)
      setStandaloneQuery('')
      setError('')
      const response = await queryClient.fetchQuery({
        queryKey: ['sessionMessages', sessionId],
        queryFn: () => fetchSessionMessages(sessionId),
      })
      setMessages(response.messages.map(mapSessionMessage))
    },
    [queryClient],
  )

  const startNewSession = useCallback(() => {
    if (abortRef.current) {
      return
    }
    setActiveSessionId(null)
    setMessages([])
    setStandaloneQuery('')
    setError('')
    setStatusText('')
    setRetrievalScore(null)
    setRetrievalAttempts(1)
  }, [])

  const clearAllSessions = useCallback(async () => {
    await deleteAllMutation.mutateAsync()
  }, [deleteAllMutation])

  const removeSessions = useCallback(
    async (sessionIds: number[]) => {
      const targetIds = Array.from(new Set(sessionIds.filter((item) => item > 0)))
      if (targetIds.length === 0) {
        return
      }
      await deleteSelectedMutation.mutateAsync(targetIds)
    },
    [deleteSelectedMutation],
  )

  const runKeywordSearch = useCallback(
    async (keyword: string) => {
      const trimmed = keyword.trim()
      if (!trimmed) {
        setSearchResults([])
        return
      }
      await searchMutation.mutateAsync(trimmed)
    },
    [searchMutation],
  )

  return {
    activeSessionId,
    messages,
    standaloneQuery,
    error,
    statusText,
    retrievalScore,
    retrievalAttempts,
    isStreaming: sendMutation.isPending,
    sessions: sessionsQuery.data ?? [],
    sessionsLoading: sessionsQuery.isLoading,
    sessionMessagesLoading: sessionMessagesQuery.isLoading,
    searchResults,
    searching: searchMutation.isPending,
    sendMessage,
    cancelStreaming,
    selectSession,
    startNewSession,
    clearAllSessions,
    runKeywordSearch,
    removeSessions,
    deletingSessions: deleteSelectedMutation.isPending || deleteAllMutation.isPending,
  }
}
