from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from collections.abc import AsyncGenerator
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.chat import ChatRequest, ChatResponse
from app.core.config import settings
from app.db.models import ChatMessage, ChatSession
from app.db.session import get_db
from app.rag.generate.engine import (
    generate_structured_answer,
    rewrite_standalone_query,
    split_stream_text,
    summarize_history_and_keywords,
)
from app.rag.pipeline import retrieve_with_retry

logger = logging.getLogger(__name__)
router = APIRouter(prefix=settings.api_prefix, tags=["chat"])


def _sse_event(event: str, payload: dict) -> str:
    body = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {body}\n\n"


def _new_trace_id() -> str:
    timestamp = datetime.now(tz=timezone.utc).strftime("%H%M%S")
    return f"{timestamp}-{uuid4().hex[:8]}"


async def _get_or_create_session(db: AsyncSession, session_id: int | None) -> ChatSession:
    if session_id is not None:
        session = await db.get(ChatSession, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session_id not found")
        return session

    session = ChatSession()
    db.add(session)
    await db.flush()
    await db.commit()
    await db.refresh(session)
    return session


async def _load_history(db: AsyncSession, session_id: int) -> list[dict[str, str]]:
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(desc(ChatMessage.id))
        .limit(settings.max_history_messages)
    )
    rows = list((await db.scalars(stmt)).all())
    rows.reverse()

    return [{"role": row.role, "content": row.content} for row in rows]


def _update_session_title(session: ChatSession, user_message: str) -> None:
    if session.title == "RAG Chatbot":
        session.title = user_message.strip()[:60] or session.title


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: AsyncSession = Depends(get_db)) -> ChatResponse:
    trace_id = _new_trace_id()
    try:
        logger.info(
            "[trace:%s] chat.request mode=sync session_id=%s message=%r",
            trace_id,
            request.session_id,
            request.message[:220],
        )
        session = await _get_or_create_session(db, request.session_id)
        logger.info("[trace:%s] chat.session resolved_session_id=%s", trace_id, session.id)
        history = await _load_history(db, session.id)
        logger.info("[trace:%s] chat.history turns=%s", trace_id, len(history))
        history_meta = await summarize_history_and_keywords(history)

        standalone_query = await rewrite_standalone_query(
            request.message,
            history,
            history_summary=history_meta.get("summary", ""),
            history_keywords=history_meta.get("keywords", []),
        )
        logger.info("[trace:%s] chat.query standalone=%r", trace_id, standalone_query)

        _update_session_title(session, request.message)
        db.add(ChatMessage(session_id=session.id, role="user", content=request.message))

        chunks, retrieval_meta = await retrieve_with_retry(
            standalone_query,
            db,
            history,
            history_meta=history_meta,
            trace_id=trace_id,
        )
        result = await generate_structured_answer(standalone_query, chunks, trace_id=trace_id)

        db.add(
            ChatMessage(
                session_id=session.id,
                role="assistant",
                content=result["answer"],
                citations_json={
                    "references": result["references"],
                    "confidence": result["confidence"],
                    "standalone_query": standalone_query,
                    "retrieval": retrieval_meta,
                    "trace_id": trace_id,
                },
            )
        )

        await db.commit()
        logger.info(
            "[trace:%s] chat.response session_id=%s confidence=%s references=%s retrieval_score=%s",
            trace_id,
            session.id,
            result["confidence"],
            len(result["references"]),
            retrieval_meta.get("best_score"),
        )

        attempts = retrieval_meta.get("attempts", [])
        return ChatResponse(
            session_id=session.id,
            standalone_query=standalone_query,
            answer=result["answer"],
            references=result["references"],
            confidence=result["confidence"],
            retrieval_score=retrieval_meta.get("best_score"),
            retrieval_attempts=len(attempts) if attempts else 1,
        )
    except HTTPException:
        raise
    except Exception as exc:
        await db.rollback()
        logger.exception("[trace:%s] chat.error %s", trace_id, exc)
        raise HTTPException(status_code=500, detail="Failed to generate response") from exc


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, db: AsyncSession = Depends(get_db)) -> StreamingResponse:
    trace_id = _new_trace_id()

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            logger.info(
                "[trace:%s] chat.request mode=stream session_id=%s message=%r",
                trace_id,
                request.session_id,
                request.message[:220],
            )
            session = await _get_or_create_session(db, request.session_id)
            logger.info("[trace:%s] chat.session resolved_session_id=%s", trace_id, session.id)
            yield _sse_event("session", {"session_id": session.id})

            history = await _load_history(db, session.id)
            logger.info("[trace:%s] chat.history turns=%s", trace_id, len(history))
            history_meta = await summarize_history_and_keywords(history)
            standalone_query = await rewrite_standalone_query(
                request.message,
                history,
                history_summary=history_meta.get("summary", ""),
                history_keywords=history_meta.get("keywords", []),
            )
            logger.info("[trace:%s] chat.query standalone=%r", trace_id, standalone_query)
            yield _sse_event("meta", {"standalone_query": standalone_query})

            _update_session_title(session, request.message)
            db.add(ChatMessage(session_id=session.id, role="user", content=request.message))

            yield _sse_event("status", {"stage": "retrieval"})
            chunks, retrieval_meta = await retrieve_with_retry(
                standalone_query,
                db,
                history,
                history_meta=history_meta,
                trace_id=trace_id,
            )
            yield _sse_event(
                "retrieval",
                {
                    "best_score": retrieval_meta.get("best_score"),
                    "attempts": retrieval_meta.get("attempts", []),
                },
            )

            yield _sse_event("status", {"stage": "generation"})
            result = await generate_structured_answer(standalone_query, chunks, trace_id=trace_id)

            db.add(
                ChatMessage(
                    session_id=session.id,
                    role="assistant",
                    content=result["answer"],
                    citations_json={
                        "references": result["references"],
                        "confidence": result["confidence"],
                        "standalone_query": standalone_query,
                        "retrieval": retrieval_meta,
                        "trace_id": trace_id,
                    },
                )
            )
            await db.commit()
            logger.info(
                "[trace:%s] chat.response session_id=%s confidence=%s references=%s retrieval_score=%s",
                trace_id,
                session.id,
                result["confidence"],
                len(result["references"]),
                retrieval_meta.get("best_score"),
            )

            for piece in split_stream_text(result["answer"]):
                yield _sse_event("delta", {"text": piece})
                await asyncio.sleep(0.01)

            attempts = retrieval_meta.get("attempts", [])
            yield _sse_event(
                "final",
                {
                    "session_id": session.id,
                    "standalone_query": standalone_query,
                    "answer": result["answer"],
                    "references": result["references"],
                    "confidence": result["confidence"],
                    "retrieval_score": retrieval_meta.get("best_score"),
                    "retrieval_attempts": len(attempts) if attempts else 1,
                },
            )

        except HTTPException as exc:
            await db.rollback()
            yield _sse_event("error", {"detail": exc.detail})
        except Exception as exc:
            await db.rollback()
            logger.exception("[trace:%s] chat.stream.error %s", trace_id, exc)
            yield _sse_event("error", {"detail": "Failed to stream response"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
