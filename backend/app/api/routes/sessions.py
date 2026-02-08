from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import delete, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.chat import ChatSearchResult, SessionMessage, SessionMessagesResponse, SessionSummary
from app.core.config import settings
from app.db.models import ChatMessage, ChatSession
from app.db.session import get_db

router = APIRouter(prefix=settings.api_prefix, tags=["sessions"])


def _build_session_message(row: ChatMessage) -> SessionMessage:
    citations = row.citations_json or {}
    references = citations.get("references") or []

    return SessionMessage(
        id=row.id,
        role=row.role,
        content=row.content,
        references=references,
        confidence=citations.get("confidence"),
        standalone_query=citations.get("standalone_query"),
        created_at=row.created_at,
    )


@router.get("/sessions", response_model=list[SessionSummary])
async def list_sessions(db: AsyncSession = Depends(get_db)) -> list[SessionSummary]:
    sessions = list((await db.scalars(select(ChatSession).order_by(desc(ChatSession.id)))).all())

    if not sessions:
        return []

    session_ids = [session.id for session in sessions]
    message_rows = list(
        (
            await db.scalars(
                select(ChatMessage)
                .where(ChatMessage.session_id.in_(session_ids))
                .order_by(ChatMessage.session_id, ChatMessage.id)
            )
        ).all()
    )

    by_session: dict[int, list[ChatMessage]] = {session_id: [] for session_id in session_ids}
    for message in message_rows:
        by_session.setdefault(message.session_id, []).append(message)

    summaries: list[SessionSummary] = []
    for session in sessions:
        session_messages = by_session.get(session.id, [])
        last_message = session_messages[-1] if session_messages else None

        summaries.append(
            SessionSummary(
                session_id=session.id,
                title=session.title,
                last_message_preview=(last_message.content[:80] if last_message else ""),
                last_message_at=(last_message.created_at if last_message else session.created_at),
                message_count=len(session_messages),
            )
        )

    return summaries


@router.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
async def get_session_messages(session_id: int, db: AsyncSession = Depends(get_db)) -> SessionMessagesResponse:
    session = await db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    rows = list(
        (
            await db.scalars(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.id)
            )
        ).all()
    )

    return SessionMessagesResponse(session_id=session_id, messages=[_build_session_message(row) for row in rows])


@router.delete("/sessions")
async def delete_all_sessions(db: AsyncSession = Depends(get_db)) -> dict:
    await db.execute(delete(ChatSession))
    await db.commit()
    return {"status": "ok"}


@router.get("/sessions/search", response_model=list[ChatSearchResult])
async def search_chat_messages(
    keyword: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(default=30, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> list[ChatSearchResult]:
    pattern = f"%{keyword.strip()}%"
    rows = list(
        (
            await db.scalars(
                select(ChatMessage)
                .where(ChatMessage.content.ilike(pattern))
                .order_by(desc(ChatMessage.created_at))
                .limit(limit)
            )
        ).all()
    )

    results: list[ChatSearchResult] = []
    for row in rows:
        content = row.content
        idx = content.lower().find(keyword.lower())
        if idx < 0:
            snippet = content[:140]
        else:
            start = max(0, idx - 40)
            end = min(len(content), idx + len(keyword) + 80)
            snippet = content[start:end]

        results.append(
            ChatSearchResult(
                session_id=row.session_id,
                message_id=row.id,
                role=row.role,
                snippet=snippet,
                created_at=row.created_at,
            )
        )

    return results
