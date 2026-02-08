from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: int | None = None


class ReferenceItem(BaseModel):
    chunk_id: int
    filename: str
    page: int
    quote: str
    source_excerpt: str | None = None


class ChatResponse(BaseModel):
    session_id: int
    standalone_query: str
    answer: str
    references: list[ReferenceItem]
    confidence: str
    retrieval_score: float | None = None
    retrieval_attempts: int = 1


class SessionSummary(BaseModel):
    session_id: int
    title: str
    last_message_preview: str
    last_message_at: datetime | None
    message_count: int


class SessionMessage(BaseModel):
    id: int
    role: str
    content: str
    references: list[ReferenceItem] = Field(default_factory=list)
    confidence: str | None = None
    standalone_query: str | None = None
    created_at: datetime


class SessionMessagesResponse(BaseModel):
    session_id: int
    messages: list[SessionMessage]


class ChatSearchResult(BaseModel):
    session_id: int
    message_id: int
    role: str
    snippet: str
    created_at: datetime


class UploadResponse(BaseModel):
    filename: str
    document_id: int
    year: str | None
    quarter: str | None
    pages: int
    chunks: int
    token_estimate: int | None = None
    saved_path: str
