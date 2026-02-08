import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, FormEvent, KeyboardEvent } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ChevronLeft,
  FileSearch,
  FileUp,
  Menu,
  MessageSquarePlus,
  Search,
  SendHorizonal,
  Square,
  Trash2,
  X,
} from "lucide-react";

import { useChat } from "../../hooks/useChat";
import { useUploadPdf } from "../../hooks/useUploadPdf";
import type { ReferenceItem } from "../../types/chat";

import "./ChatInterface.css";

function ThinkingDots() {
  const [dotCount, setDotCount] = useState(1);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setDotCount((prev) => (prev >= 3 ? 1 : prev + 1));
    }, 360);

    return () => {
      window.clearInterval(timer);
    };
  }, []);

  return (
    <span className="thinking-indicator-dots">{".".repeat(dotCount)}</span>
  );
}

function formatRelativeTime(dateText: string | null): string {
  if (!dateText) {
    return "방금 전";
  }

  const date = new Date(dateText);
  const now = new Date();
  const diffSec = Math.max(
    1,
    Math.floor((now.getTime() - date.getTime()) / 1000),
  );

  if (diffSec < 60) {
    return `${diffSec}초 전`;
  }

  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) {
    return `${diffMin}분 전`;
  }

  const diffHour = Math.floor(diffMin / 60);
  if (diffHour < 24) {
    return `${diffHour}시간 전`;
  }

  const diffDay = Math.floor(diffHour / 24);
  return `${diffDay}일 전`;
}

export default function ChatInterface() {
  const [input, setInput] = useState("");
  const [searchKeyword, setSearchKeyword] = useState("");
  const [selectedReference, setSelectedReference] =
    useState<ReferenceItem | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const chatLogRef = useRef<HTMLElement | null>(null);

  const {
    activeSessionId,
    messages,
    error,
    statusText,
    isStreaming,
    sessions,
    sessionsLoading,
    searchResults,
    searching,
    sendMessage,
    cancelStreaming,
    selectSession,
    startNewSession,
    clearAllSessions,
    runKeywordSearch,
  } = useChat();

  const uploadMutation = useUploadPdf();

  const hasMessages = messages.length > 0;

  const visibleStatus = useMemo(() => {
    if (statusText) {
      return statusText;
    }
    if (isStreaming) {
      return "생각 중...";
    }
    return "";
  }, [statusText, isStreaming]);

  useEffect(() => {
    const container = chatLogRef.current;
    if (!container) {
      return;
    }
    container.scrollTop = container.scrollHeight;
  }, [messages, visibleStatus]);

  const handleSend = async (event?: FormEvent) => {
    event?.preventDefault();
    if (!input.trim() || isStreaming) {
      return;
    }

    const next = input;
    setInput("");
    try {
      await sendMessage(next);
    } catch {
      // Error state is handled inside useChat.
    }
  };

  const handleEnter = async (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      await handleSend();
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";

    if (!file) {
      return;
    }

    try {
      await uploadMutation.mutateAsync(file);
    } catch {
      // Mutation error text is rendered in UI.
    }
  };

  const handleSearchSubmit = async (event: FormEvent) => {
    event.preventDefault();
    await runKeywordSearch(searchKeyword);
  };

  return (
    <div
      className={`chat-layout ${isSidebarOpen ? "has-sidebar" : ""} ${selectedReference ? "has-reference" : ""}`}
    >
      {isSidebarOpen && (
        <button
          type="button"
          className="sidebar-backdrop"
          aria-label="사이드바 닫기"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {isSidebarOpen && (
        <aside className="chat-sidebar">
          <div className="sidebar-top">
            <div className="sidebar-title-row">
              <button
                type="button"
                className="brand-btn"
                onClick={startNewSession}
                disabled={isStreaming}
              >
                RAG Chatbot
              </button>
              <button
                type="button"
                className="icon-btn"
                aria-label="사이드바 숨기기"
                onClick={() => setIsSidebarOpen(false)}
              >
                <ChevronLeft size={14} />
              </button>
            </div>

            <form className="sidebar-search" onSubmit={handleSearchSubmit}>
              <label htmlFor="chat-search">전체 채팅 검색</label>
              <div className="sidebar-search-row">
                <input
                  id="chat-search"
                  value={searchKeyword}
                  onChange={(event) => setSearchKeyword(event.target.value)}
                  placeholder="키워드 입력"
                />
                <button type="submit" className="icon-btn" disabled={searching}>
                  <Search size={14} />
                </button>
              </div>
            </form>

            {searchResults.length > 0 && (
              <div className="search-results">
                {searchResults.slice(0, 10).map((result) => (
                  <button
                    key={`${result.session_id}-${result.message_id}`}
                    type="button"
                    className="search-item"
                    onClick={() => {
                      void selectSession(result.session_id);
                    }}
                  >
                    <div>{result.snippet}</div>
                    <span>{formatRelativeTime(result.created_at)}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="sidebar-sessions">
            <button
              type="button"
              className="sidebar-btn"
              onClick={startNewSession}
              disabled={isStreaming}
            >
              <MessageSquarePlus size={15} />새 채팅
            </button>
            <h2>채팅 목록</h2>
            {sessionsLoading ? (
              <p className="sidebar-muted">불러오는 중...</p>
            ) : sessions.length === 0 ? (
              <p className="sidebar-muted">저장된 채팅이 없습니다.</p>
            ) : (
              sessions.map((session) => (
                <button
                  key={session.session_id}
                  type="button"
                  className={`session-item ${activeSessionId === session.session_id ? "active" : ""}`}
                  onClick={() => {
                    void selectSession(session.session_id);
                  }}
                  disabled={isStreaming}
                >
                  <strong>{session.title}</strong>
                  <p>{session.last_message_preview || "대화 시작"}</p>
                  <span>{formatRelativeTime(session.last_message_at)}</span>
                </button>
              ))
            )}
          </div>

          <div className="sidebar-bottom">
            <button
              type="button"
              className="danger-btn"
              onClick={() => void clearAllSessions()}
              disabled={isStreaming}
            >
              <Trash2 size={15} />
              전체 채팅 삭제
            </button>
          </div>
        </aside>
      )}

      <section className="chat-main">
        <header className="chat-header">
          <div>
            <h2>RAG Chatbot</h2>
            <p>PDF 기반 질의응답 챗봇</p>
          </div>
          <div className="chat-header-actions">
            <button
              type="button"
              className="ghost-btn"
              onClick={() => setIsSidebarOpen((prev) => !prev)}
              aria-label={isSidebarOpen ? "사이드바 숨기기" : "사이드바 열기"}
            >
              <Menu size={16} />
              {isSidebarOpen ? "사이드바 숨기기" : "사이드바 열기"}
            </button>
            <button
              type="button"
              className="ghost-btn"
              onClick={handleUploadClick}
              disabled={uploadMutation.isPending}
            >
              <FileUp size={16} />
              {uploadMutation.isPending ? "업로드 중..." : "PDF 업로드"}
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="application/pdf"
              className="hidden-input"
              onChange={handleFileChange}
            />
          </div>
        </header>

        <main className="chat-log" aria-live="polite" ref={chatLogRef}>
          {!hasMessages ? (
            <div className="empty-state">
              삼성전자의 분기별 실적, 사업부별 매출/영업이익, 전분기 대비 증감
              등 질문을 입력하세요.
            </div>
          ) : (
            messages.map((message) => (
              <article key={message.id} className={`message ${message.role}`}>
                <div className="message-body">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {message.content}
                  </ReactMarkdown>
                </div>

                {message.references && message.references.length > 0 && (
                  <div className="message-references">
                    <strong>출처</strong>
                    {message.references.map((reference) => (
                      <button
                        key={`${reference.chunk_id}-${reference.quote.slice(0, 12)}`}
                        type="button"
                        className="reference-item"
                        onClick={() => setSelectedReference(reference)}
                      >
                        <FileSearch size={14} />
                        {reference.filename} (p.{reference.page})
                      </button>
                    ))}
                  </div>
                )}

                {message.confidence && (
                  <div className="message-confidence">
                    Confidence: {message.confidence}
                  </div>
                )}
              </article>
            ))
          )}

          {visibleStatus && (
            <div className="thinking-indicator">
              {visibleStatus.replace(/[.…]+$/g, "")}
              <ThinkingDots />
            </div>
          )}
        </main>

        {(error || uploadMutation.error) && (
          <section className="chat-error">
            {error && <div>{error}</div>}
            {uploadMutation.error && <div>{uploadMutation.error.message}</div>}
          </section>
        )}

        <form className="chat-input" onSubmit={handleSend}>
          <div className="chat-input-shell">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleEnter}
              placeholder="예시: 2024년 1분기 DS 부문 매출과 영업이익을 원문 인용과 함께 알려줘."
              rows={3}
            />
            <div className="chat-input-actions">
              <button
                type="button"
                className="ghost-btn"
                onClick={cancelStreaming}
                disabled={!isStreaming}
              >
                <Square size={16} />
                중단
              </button>
              <button
                type="submit"
                className="primary-btn"
                disabled={isStreaming || !input.trim()}
              >
                <SendHorizonal size={16} />
                전송
              </button>
            </div>
          </div>
        </form>
      </section>

      {selectedReference && (
        <aside className="reference-panel">
          <header>
            <h3>
              {selectedReference.filename} p.{selectedReference.page}
            </h3>
            <button
              type="button"
              className="icon-btn"
              onClick={() => setSelectedReference(null)}
            >
              <X size={14} />
            </button>
          </header>
          <blockquote>{selectedReference.quote}</blockquote>
          <div className="reference-panel-body">
            {selectedReference.source_excerpt ?? selectedReference.quote}
          </div>
        </aside>
      )}
    </div>
  );
}
