# backend/app/rag/pipeline.py
from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import Chunk
from app.rag.generate.engine import (
    evaluate_retrieval_quality,
    generate_multi_queries,
    summarize_history_and_keywords,
)
from app.rag.retrieve.hybrid import hybrid_search, rerank_candidates

logger = logging.getLogger(__name__)

try:
    from langchain_core.runnables import RunnableLambda
except Exception:  # pragma: no cover - optional runtime fallback
    RunnableLambda = None

TOKEN_PATTERN = re.compile(r"[0-9]+(?:\.[0-9]+)?|[A-Za-z]+|[가-힣]+")
YEAR_PATTERN = re.compile(r"(20\d{2})")
QUARTER_PATTERN = re.compile(r"(?:([1-4])\s*Q|Q\s*([1-4])|([1-4])\s*분기)", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")


def _normalize_query_plan(queries: list[str]) -> list[str]:
    cleaned: list[str] = []
    for query in queries:
        stripped = query.strip()
        if not stripped or stripped in cleaned:
            continue
        cleaned.append(stripped)
    return cleaned[: settings.multi_query_count]


def _prepare_queries_with_langchain(question: str, generated_queries: list[str]) -> list[str]:
    planned = [question, *generated_queries]
    if RunnableLambda is None:
        return _normalize_query_plan(planned)

    chain = RunnableLambda(lambda payload: payload["queries"]) | RunnableLambda(_normalize_query_plan)
    return chain.invoke({"queries": planned})


def _chunk_trace(chunk: Chunk) -> dict[str, Any]:
    meta = chunk.metadata_json or {}
    return {
        "chunk_id": chunk.id,
        "source": meta.get("source", "unknown"),
        "page": int(meta.get("page", chunk.page_number)),
        "chunk_type": meta.get("chunk_type", "unknown"),
        "year": meta.get("year", ""),
        "quarter": meta.get("quarter", ""),
    }


def _table_chunk_key(chunk: Chunk) -> tuple[str, str] | None:
    meta = chunk.metadata_json or {}
    chunk_type = str(meta.get("chunk_type", "")).lower()
    if chunk_type != "table":
        return None
    source = str(meta.get("source", "")).strip()
    page = str(meta.get("page", chunk.page_number)).strip()
    if not source or not page:
        return None
    return source, page


async def _augment_with_table_page_siblings(chunks: list[Chunk], db: AsyncSession, limit: int) -> list[Chunk]:
    if not chunks:
        return chunks

    keys: list[tuple[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    for chunk in chunks:
        key = _table_chunk_key(chunk)
        if key is None or key in seen_keys:
            continue
        seen_keys.add(key)
        keys.append(key)

    if not keys:
        return chunks

    siblings_by_key: dict[tuple[str, str], list[Chunk]] = {}
    for source, page in keys:
        stmt = (
            select(Chunk)
            .where(Chunk.metadata_json["chunk_type"].astext == "table")
            .where(Chunk.metadata_json["source"].astext == source)
            .where(Chunk.metadata_json["page"].astext == page)
            .order_by(Chunk.id)
        )
        siblings = (await db.execute(stmt)).scalars().all()
        siblings_by_key[(source, page)] = siblings

    augmented: list[Chunk] = []
    seen_ids: set[int] = set()
    for chunk in chunks:
        if chunk.id not in seen_ids:
            seen_ids.add(chunk.id)
            augmented.append(chunk)

        key = _table_chunk_key(chunk)
        if key is None:
            continue
        for sibling in siblings_by_key.get(key, []):
            if sibling.id in seen_ids:
                continue
            seen_ids.add(sibling.id)
            augmented.append(sibling)

    if limit > 0 and len(augmented) > limit:
        return augmented[:limit]
    return augmented


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _extract_temporal_hints(query: str) -> tuple[str | None, str | None]:
    quarter = QUARTER_PATTERN.search(query)
    quarter_value: str | None = None
    if quarter:
        quarter_digit = quarter.group(1) or quarter.group(2) or quarter.group(3)
        quarter_value = f"{quarter_digit}Q" if quarter_digit else None

    year = YEAR_PATTERN.search(query)
    year_value = year.group(1) if year else None
    return year_value, quarter_value


def _cheap_retrieval_quality(question: str, chunks: list[Chunk]) -> dict[str, Any]:
    if not chunks:
        return {
            "score": 0.0,
            "reason": "cheap-heuristic:no chunks",
            "improved_query": question,
            "must_have_terms": [],
        }

    question_tokens = {token for token in _tokenize(question) if len(token) >= 2 and not token.isdigit()}
    if not question_tokens:
        question_tokens = set(_tokenize(question))

    overlap_scores: list[float] = []
    numeric_hits = 0
    year_hint, quarter_hint = _extract_temporal_hints(question)
    temporal_hits = 0
    temporal_eligible = 0

    for chunk in chunks:
        content = chunk.content[:1200]
        chunk_tokens = set(_tokenize(content))
        overlap_scores.append(len(question_tokens & chunk_tokens) / max(1, len(question_tokens)))

        if NUMBER_PATTERN.search(content):
            numeric_hits += 1

        if year_hint or quarter_hint:
            meta = chunk.metadata_json or {}
            temporal_eligible += 1
            year_match = (not year_hint) or str(meta.get("year", "")).strip() == year_hint
            quarter_match = (not quarter_hint) or str(meta.get("quarter", "")).strip().upper() == quarter_hint
            if year_match and quarter_match:
                temporal_hits += 1

    overlap_score = max(overlap_scores) if overlap_scores else 0.0
    numeric_ratio = numeric_hits / max(1, len(chunks))

    if year_hint or quarter_hint:
        temporal_ratio = temporal_hits / max(1, temporal_eligible)
        score = (overlap_score * 0.5) + (numeric_ratio * 0.2) + (temporal_ratio * 0.3)
    else:
        temporal_ratio = 0.0
        score = (overlap_score * 0.7) + (numeric_ratio * 0.3)

    must_have_terms = list(question_tokens)[:4]
    reason = (
        "cheap-heuristic:"
        f" overlap={overlap_score:.3f}"
        f" numeric_ratio={numeric_ratio:.3f}"
        f" temporal_ratio={temporal_ratio:.3f}"
    )
    return {
        "score": max(0.0, min(1.0, score)),
        "reason": reason,
        "improved_query": question,
        "must_have_terms": must_have_terms,
    }


async def retrieve_with_retry(
    question: str,
    db: AsyncSession,
    history: list[dict[str, str]],
    history_meta: dict | None = None,
    trace_id: str | None = None,
) -> tuple[list[Chunk], dict]:
    if history_meta is None:
        history_meta = await summarize_history_and_keywords(history)

    base_queries = await generate_multi_queries(
        question,
        history_summary=history_meta.get("summary", ""),
        history_keywords=history_meta.get("keywords", []),
    )
    base_queries = _prepare_queries_with_langchain(question, base_queries)
    if trace_id:
        logger.info(
            "[trace:%s] retrieval.plan question=%r queries=%s max_attempts=%s threshold=%.2f",
            trace_id,
            question,
            base_queries,
            settings.retrieval_max_attempts,
            settings.retrieval_quality_threshold,
        )

    attempts: list[dict] = []
    candidate_queries = list(base_queries)
    best_chunks: list[Chunk] = []
    best_score = -1.0
    embedding_cache: dict[str, list[float]] = {}

    for attempt in range(1, settings.retrieval_max_attempts + 1):
        if trace_id:
            logger.info("[trace:%s] retrieval.attempt.start attempt=%s queries=%s", trace_id, attempt, candidate_queries)

        query_results: list[list[Chunk]] = []
        for query_index, query in enumerate(candidate_queries):
            items = await hybrid_search(
                query,
                db,
                vector_top_k=settings.vector_top_k,
                keyword_top_k=settings.keyword_top_k,
                fused_top_k=settings.fused_top_k,
                final_top_k=settings.fused_top_k,
                trace_id=trace_id,
                attempt=attempt,
                query_index=query_index + 1,
                do_rerank=False,
                embedding_cache=embedding_cache,
            )
            query_results.append(items)

        flattened: list[Chunk] = []
        query_stats: list[dict[str, Any]] = []
        for query_index, items in enumerate(query_results):
            flattened.extend(items)
            query_stats.append(
                {
                    "query": candidate_queries[query_index],
                    "result_count": len(items),
                    "top_chunks": [_chunk_trace(chunk) for chunk in items[:3]],
                }
            )

        dedup_ids = {chunk.id for chunk in flattened}
        if trace_id:
            logger.info(
                "[trace:%s] retrieval.attempt.aggregate attempt=%s flattened=%s dedup=%s",
                trace_id,
                attempt,
                len(flattened),
                len(dedup_ids),
            )

        reranked = await rerank_candidates(question, flattened, settings.max_context_chunks)
        # Keep table page siblings even if it slightly exceeds top-k, to avoid losing
        # split header/body table chunks that must be read together.
        reranked = await _augment_with_table_page_siblings(reranked, db, limit=0)

        evaluation_mode = "llm"
        cheap_eval: dict[str, Any] | None = None
        if attempt > 1:
            cheap_eval = _cheap_retrieval_quality(question, reranked)
            cheap_score = float(cheap_eval.get("score", 0.0))
            if trace_id:
                logger.info(
                    "[trace:%s] retrieval.attempt.cheap_eval attempt=%s score=%.4f reason=%r",
                    trace_id,
                    attempt,
                    cheap_score,
                    cheap_eval.get("reason", ""),
                )

            if cheap_score >= settings.retrieval_quality_threshold:
                evaluation_mode = "cheap-heuristic"
                evaluation = cheap_eval
            else:
                evaluation_mode = "llm-compact"
                evaluation = await evaluate_retrieval_quality(
                    question,
                    reranked[:6],
                    max_context_chars=min(4500, settings.max_context_chars),
                    max_chunks=6,
                    max_chars_per_chunk=220,
                )
                evaluation["reason"] = (
                    f"{evaluation.get('reason', '')} | cheap_score={cheap_score:.3f}"
                    if evaluation.get("reason", "")
                    else f"cheap_score={cheap_score:.3f}"
                )
        else:
            evaluation = await evaluate_retrieval_quality(question, reranked)
        score = float(evaluation.get("score", 0.0))
        selected_chunks = [_chunk_trace(chunk) for chunk in reranked]

        if trace_id:
            logger.info(
                "[trace:%s] retrieval.attempt.eval attempt=%s mode=%s score=%.4f reason=%r improved_query=%r must_have_terms=%s selected_chunks=%s",
                trace_id,
                attempt,
                evaluation_mode,
                score,
                evaluation.get("reason", ""),
                evaluation.get("improved_query", ""),
                evaluation.get("must_have_terms", []),
                selected_chunks,
            )

        attempts.append(
            {
                "attempt": attempt,
                "queries": list(candidate_queries),
                "score": score,
                "evaluation_mode": evaluation_mode,
                "cheap_score": float(cheap_eval.get("score", 0.0)) if cheap_eval else None,
                "reason": evaluation.get("reason", ""),
                "query_stats": query_stats,
                "selected_chunks": selected_chunks,
            }
        )

        if score > best_score:
            best_score = score
            best_chunks = reranked
            if trace_id:
                logger.info(
                    "[trace:%s] retrieval.best.updated attempt=%s best_score=%.4f chunk_ids=%s",
                    trace_id,
                    attempt,
                    best_score,
                    [chunk.id for chunk in best_chunks],
                )

        if score >= settings.retrieval_quality_threshold and reranked:
            if trace_id:
                logger.info("[trace:%s] retrieval.attempt.accepted attempt=%s score=%.4f", trace_id, attempt, score)
            break

        improved_query = str(evaluation.get("improved_query", "")).strip()
        must_have_terms = [str(item).strip() for item in evaluation.get("must_have_terms", []) if str(item).strip()]

        next_queries = [question]
        if improved_query:
            next_queries.append(improved_query)

        for term in must_have_terms:
            expanded = f"{question} {term}".strip()
            if expanded not in next_queries:
                next_queries.append(expanded)

        for existing in base_queries:
            if existing not in next_queries:
                next_queries.append(existing)

        candidate_queries = next_queries[: settings.multi_query_count]
        if trace_id:
            logger.info("[trace:%s] retrieval.attempt.next_queries attempt=%s next=%s", trace_id, attempt, candidate_queries)

    best_chunk_trace = [_chunk_trace(chunk) for chunk in best_chunks]
    if trace_id:
        logger.info(
            "[trace:%s] retrieval.done best_score=%.4f best_chunks=%s attempts=%s",
            trace_id,
            best_score,
            best_chunk_trace,
            len(attempts),
        )
    return best_chunks, {
        "history_summary": history_meta,
        "attempts": attempts,
        "best_score": best_score,
        "best_chunks": best_chunk_trace,
    }
