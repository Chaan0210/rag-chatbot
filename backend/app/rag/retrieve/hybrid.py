# backend/app/rag/retrieve/hybrid.py
from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi
from sqlalchemy import and_, case, func, literal, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import Chunk, Document

logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=settings.openai_api_key)

TOKEN_PATTERN = re.compile(r"[0-9]+(?:\.[0-9]+)?|[A-Za-z]+|[가-힣]+")
YEAR_PATTERN = re.compile(r"(20\d{2})")
QUARTER_PATTERN = re.compile(r"(?:([1-4])\s*Q|Q\s*([1-4])|([1-4])\s*분기)", re.IGNORECASE)
QUARTER_YEAR_PATTERN = re.compile(
    r"(?:([1-4])\s*Q\s*['’]?\s*(\d{2,4})|Q\s*([1-4])\s*['’]?\s*(\d{2,4}))",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")

DS_TERMS = ("ds", "device solutions", "디바이스솔루션", "반도체")
DS_WORD_PATTERN = re.compile(r"(?<![0-9A-Za-z가-힣])ds(?![0-9A-Za-z가-힣])", re.IGNORECASE)
REVENUE_TERMS = ("매출", "매출액", "revenue", "sales")
OPERATING_PROFIT_TERMS = ("영업이익", "영업손익", "operating profit", "operating income")
KEYWORD_PRIORITY_TERMS = {
    "매출",
    "매출액",
    "revenue",
    "sales",
    "영업이익",
    "영업손익",
    "순이익",
    "profit",
    "income",
    "margin",
    "qoq",
    "yoy",
}
KEYWORD_ENTITY_TERMS = {"ds", "mx", "vd", "da", "하만", "메모리", "반도체"}

RRF_K = 60
TRGM_SIMILARITY_FLOOR = 0.05
TRGM_TOTAL_SIMILARITY_FLOOR = 0.12
RERANKER_MODEL = settings.reranker_model
_reranker: Any | None = None
_reranker_failed = False


def _chunk_summary(chunk: Chunk) -> dict[str, Any]:
    meta = chunk.metadata_json or {}
    return {
        "chunk_id": chunk.id,
        "source": meta.get("source", "unknown"),
        "page": int(meta.get("page", chunk.page_number)),
        "chunk_type": meta.get("chunk_type", "unknown"),
    }


def _preview_scored(scored: list[tuple[Chunk, float]], score_key: str, top_n: int = 5) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for chunk, score in scored[:top_n]:
        item = _chunk_summary(chunk)
        item[score_key] = round(float(score), 6)
        preview.append(item)
    return preview


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _keyword_terms(query: str, max_terms: int = 4) -> list[str]:
    tokens = _tokenize(query)
    terms: list[str] = []
    seen: set[str] = set()

    def add_term(token: str) -> None:
        if len(token) < 2:
            return
        if token.isdigit():
            return
        if token in seen:
            return
        seen.add(token)
        terms.append(token)

    for token in tokens:
        if token in KEYWORD_PRIORITY_TERMS:
            add_term(token)
        if len(terms) >= max_terms:
            return terms[:max_terms]

    for token in tokens:
        if token in KEYWORD_ENTITY_TERMS:
            add_term(token)
        if len(terms) >= max_terms:
            return terms[:max_terms]

    for token in tokens:
        add_term(token)
        if len(terms) >= max_terms:
            break
    return terms


def _content_without_metadata_header(content: str) -> str:
    lines = content.splitlines()
    if not lines:
        return ""
    if lines[0].startswith("[METADATA]"):
        return "\n".join(lines[1:]).strip()
    return content.strip()


def _extract_temporal_hints(query: str) -> tuple[str | None, str | None]:
    quarter = QUARTER_PATTERN.search(query)
    quarter_value: str | None = None
    if quarter:
        quarter_digit = quarter.group(1) or quarter.group(2) or quarter.group(3)
        quarter_value = f"{quarter_digit}Q" if quarter_digit else None

    year_value: str | None = None
    quarter_year_matches = list(QUARTER_YEAR_PATTERN.finditer(query))
    if quarter_year_matches:
        first = quarter_year_matches[0]
        year_token = first.group(2) or first.group(4) or ""
        if len(year_token) == 2:
            year_value = f"20{year_token}"
        elif len(year_token) == 4:
            year_value = year_token

    if not year_value:
        year = YEAR_PATTERN.search(query)
        if year:
            year_value = year.group(1)

    return year_value, quarter_value


def _quarter_terms(quarter: str) -> list[str]:
    normalized = quarter.strip().upper()
    if not normalized:
        return []
    digit = re.sub(r"[^1-4]", "", normalized)
    if not digit:
        return [normalized]
    q = digit[0]
    return [f"{q}Q", f"Q{q}", f"{q}분기", f"{q} 분기"]


def _apply_temporal_filters(stmt: Any, year: str | None, quarter: str | None) -> Any:
    if year:
        year_clauses = [
            Chunk.metadata_json["year"].astext == year,
            Chunk.metadata_json["source"].astext.ilike(f"%{year}%"),
            Chunk.content.ilike(f"%{year}%"),
        ]
        stmt = stmt.where(or_(*year_clauses))
    if quarter:
        quarter_terms = _quarter_terms(quarter)
        quarter_clauses = [func.upper(Chunk.metadata_json["quarter"].astext) == quarter]
        for term in quarter_terms:
            quarter_clauses.append(Chunk.metadata_json["source"].astext.ilike(f"%{term}%"))
            quarter_clauses.append(Chunk.content.ilike(f"%{term}%"))
        stmt = stmt.where(or_(*quarter_clauses))
    return stmt


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    needs_ds_check = any(term.lower() == "ds" for term in terms)
    ds_hit = False
    if needs_ds_check:
        tokens = set(_tokenize(lowered))
        ds_hit = "ds" in tokens or bool(DS_WORD_PATTERN.search(lowered))

    for term in terms:
        normalized = term.lower()
        if normalized == "ds":
            if ds_hit:
                return True
            continue
        if normalized in lowered:
            return True
    return False


def _is_numeric_focus_query(query: str) -> bool:
    lowered = query.lower()
    has_number = bool(NUMBER_PATTERN.search(query))
    metric_signal = (
        _contains_any(lowered, REVENUE_TERMS + OPERATING_PROFIT_TERMS)
        or any(term in lowered for term in ("순이익", "마진", "margin", "roe", "ebitda", "증감", "qoq", "yoy", "%"))
    )
    if metric_signal:
        return True

    if "실적" in lowered:
        return has_number or any(term in lowered for term in ("매출", "이익", "손익", "마진"))

    return False


def _query_intent(query: str) -> dict[str, Any]:
    lowered = query.lower()
    year, quarter = _extract_temporal_hints(query)
    return {
        "year": year,
        "quarter": quarter,
        "needs_ds": _contains_any(lowered, DS_TERMS),
        "needs_revenue": _contains_any(lowered, REVENUE_TERMS),
        "needs_operating_profit": _contains_any(lowered, OPERATING_PROFIT_TERMS),
        "numeric_focus": _is_numeric_focus_query(query),
    }


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if abs(high - low) < 1e-9:
        return [0.5 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        return vector
    return [value / norm for value in vector]


def _heuristic_chunk_score(query_tokens: set[str], chunk: Chunk, intent: dict[str, Any]) -> float:
    meta = chunk.metadata_json or {}
    body = _content_without_metadata_header(chunk.content)
    text = body.lower()
    chunk_tokens = set(_tokenize(body))
    overlap = len(query_tokens & chunk_tokens) / max(1, len(query_tokens))

    score = overlap * 4.0
    if intent["year"] and str(meta.get("year", "")).strip() == intent["year"]:
        score += 1.5
    if intent["quarter"] and str(meta.get("quarter", "")).strip().upper() == intent["quarter"]:
        score += 2.0
    if intent["needs_ds"] and _contains_any(text, DS_TERMS):
        score += 1.8
    if intent["needs_revenue"] and _contains_any(text, REVENUE_TERMS):
        score += 1.6
    if intent["needs_operating_profit"] and _contains_any(text, OPERATING_PROFIT_TERMS):
        score += 1.6

    chunk_type = str(meta.get("chunk_type", "")).lower()
    if chunk_type == "table":
        score += 1.2 if intent["numeric_focus"] else 0.4

    if intent["numeric_focus"]:
        numeric_count = len(NUMBER_PATTERN.findall(body))
        score += min(1.6, numeric_count / 8.0)

    return score


async def _embed_query(query: str, embedding_cache: dict[str, list[float]] | None = None) -> list[float]:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for retrieval.")

    cache_key = query.strip()
    if embedding_cache is not None and cache_key in embedding_cache:
        return embedding_cache[cache_key]

    response = await client.embeddings.create(model=settings.embedding_model, input=[query])
    embedding = _normalize_vector(response.data[0].embedding)
    if embedding_cache is not None:
        embedding_cache[cache_key] = embedding
    return embedding


async def _vector_search(
    db: AsyncSession,
    query_embedding: list[float],
    top_k: int,
    year: str | None = None,
    quarter: str | None = None,
) -> list[tuple[Chunk, float]]:
    distance_expr = Chunk.embedding.cosine_distance(query_embedding).label("distance")
    stmt = select(Chunk, distance_expr)
    stmt = _apply_temporal_filters(stmt, year, quarter)
    stmt = stmt.order_by(distance_expr).limit(top_k)
    rows = (await db.execute(stmt)).all()

    if rows or (year is None and quarter is None):
        return [(row[0], float(row[1])) for row in rows]

    # Fallback when strict temporal filter yields no rows.
    stmt = select(Chunk, distance_expr).order_by(distance_expr).limit(top_k)
    fallback_rows = (await db.execute(stmt)).all()
    return [(row[0], float(row[1])) for row in fallback_rows]


async def _recent_document_ids(db: AsyncSession) -> list[int]:
    limit = max(1, settings.keyword_recent_docs_limit)
    stmt = select(Document.id).order_by(Document.uploaded_at.desc()).limit(limit)
    return [int(item) for item in (await db.scalars(stmt)).all()]


async def _keyword_candidates(db: AsyncSession, query: str, top_k: int) -> list[Chunk]:
    year, quarter = _extract_temporal_hints(query)
    recent_doc_ids = await _recent_document_ids(db)
    pool_limit = max(top_k, settings.keyword_candidate_pool_size)
    terms = _keyword_terms(query)

    async def fetch(
        year_value: str | None,
        quarter_value: str | None,
        use_recent_docs: bool,
        primary_chunk_only: bool,
    ) -> list[Chunk]:
        def _base_stmt(include_temporal: bool) -> Any:
            stmt = select(Chunk)
            if use_recent_docs and recent_doc_ids:
                stmt = stmt.where(Chunk.document_id.in_(recent_doc_ids))
            if include_temporal:
                stmt = _apply_temporal_filters(stmt, year_value, quarter_value)
            if primary_chunk_only:
                stmt = stmt.where(Chunk.metadata_json["chunk_type"].astext.in_(("table", "slide")))
            return stmt

        async def _run(stmt: Any) -> list[Chunk]:
            if terms:
                if settings.enable_trigram_search:
                    # Korean-heavy corpora are better served by trigram similarity than 'simple' FTS.
                    similarity_exprs = [func.similarity(Chunk.content, term) for term in terms]
                    percent_filters = [Chunk.content.op("%")(term) for term in terms]
                    similarity_filters = [expr > TRGM_SIMILARITY_FLOOR for expr in similarity_exprs]

                    if (percent_filters or similarity_filters) and similarity_exprs:
                        trgm_score_expr = literal(0.0)
                        for expr in similarity_exprs:
                            trgm_score_expr = trgm_score_expr + expr
                        trgm_score = trgm_score_expr.label("lexical_score")
                        score_threshold = TRGM_TOTAL_SIMILARITY_FLOOR * max(1.0, len(terms) / 2.0)

                        broad_filter_parts: list[Any] = []
                        if percent_filters:
                            broad_filter_parts.append(or_(*percent_filters))
                        if similarity_filters:
                            broad_filter_parts.append(or_(*similarity_filters))

                        try:
                            rows = (
                                await db.execute(
                                    stmt.where(and_(or_(*broad_filter_parts), trgm_score_expr > score_threshold))
                                    .add_columns(trgm_score)
                                    .order_by(trgm_score.desc())
                                    .limit(pool_limit)
                                )
                            ).all()
                        except Exception as exc:
                            logger.warning("Trigram keyword search failed, fallback to ILIKE: %s", exc)
                            rows = []

                        if rows:
                            return [row[0] for row in rows]

                # Fallback when trigram candidates are not found.
                term_clauses = [Chunk.content.ilike(f"%{term}%") for term in terms]
                fallback_score = literal(0)
                for clause in term_clauses:
                    fallback_score = fallback_score + case((clause, 1), else_=0)
                fallback_score = fallback_score.label("lexical_score")
                fallback_rows = (
                    await db.execute(
                        stmt.where(or_(*term_clauses))
                        .add_columns(fallback_score)
                        .order_by(fallback_score.desc())
                        .limit(pool_limit)
                    )
                ).all()
                return [row[0] for row in fallback_rows]

            chunk_type_rank = case(
                (Chunk.metadata_json["chunk_type"].astext == "table", 0),
                (Chunk.metadata_json["chunk_type"].astext == "slide", 1),
                else_=2,
            )
            stmt = stmt.order_by(
                chunk_type_rank.asc(),
                Chunk.document_id.desc(),
                Chunk.page_number.asc(),
                Chunk.chunk_index.asc(),
            ).limit(pool_limit)
            return list((await db.scalars(stmt)).all())

        has_temporal = bool(year_value or quarter_value)
        rows = await _run(_base_stmt(include_temporal=has_temporal))
        if rows or not has_temporal:
            return rows

        # Temporal metadata can be missing in filenames; retry without temporal filter.
        return await _run(_base_stmt(include_temporal=False))

    attempts: list[tuple[str | None, str | None, bool, bool]] = [
        (year, quarter, True, True),
        (year, quarter, True, False),
    ]
    if year and quarter:
        attempts.extend(
            [
                (year, None, True, True),
                (year, None, True, False),
            ]
        )
    attempts.extend(
        [
            (None, None, True, True),
            (None, None, True, False),
            (None, None, False, True),
            (None, None, False, False),
        ]
    )

    dedup: dict[int, Chunk] = {}
    for year_value, quarter_value, use_recent_docs, primary_chunk_only in attempts:
        rows = await fetch(year_value, quarter_value, use_recent_docs, primary_chunk_only)
        for row in rows:
            dedup[row.id] = row
        if len(dedup) >= pool_limit:
            break

    if dedup:
        return list(dedup.values())[:pool_limit]
    return []


async def _keyword_search(db: AsyncSession, query: str, top_k: int) -> list[tuple[Chunk, float]]:
    candidates = await _keyword_candidates(db, query, top_k=top_k)
    if not candidates:
        return []

    tokenized_query = _tokenize(query)
    tokenized_corpus = [_tokenize(_content_without_metadata_header(chunk.content)) for chunk in candidates]

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(candidates[idx], float(scores[idx])) for idx in top_indices]


def _rrf_fuse(
    vector_results: list[tuple[Chunk, float]],
    keyword_results: list[tuple[Chunk, float]],
    top_k: int,
) -> list[tuple[Chunk, float]]:
    score_map: dict[int, float] = defaultdict(float)
    by_id: dict[int, Chunk] = {}

    for rank, (chunk, _) in enumerate(vector_results, start=1):
        score_map[chunk.id] += 1 / (RRF_K + rank)
        by_id[chunk.id] = chunk

    for rank, (chunk, _) in enumerate(keyword_results, start=1):
        score_map[chunk.id] += 1 / (RRF_K + rank)
        by_id[chunk.id] = chunk

    fused_ids = sorted(score_map, key=score_map.get, reverse=True)[:top_k]
    return [(by_id[item_id], score_map[item_id]) for item_id in fused_ids]


def _load_reranker() -> Any | None:
    global _reranker
    global _reranker_failed

    if _reranker is not None:
        return _reranker
    if _reranker_failed:
        return None
    if not settings.enable_reranker:
        return None

    try:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANKER_MODEL)
        return _reranker
    except Exception as exc:
        _reranker_failed = True
        logger.warning("Cross-encoder load failed, fallback to RRF ordering: %s", exc)
        return None


async def _rerank(query: str, chunks: list[Chunk]) -> list[tuple[Chunk, float]]:
    if not chunks:
        return []

    model = _load_reranker()
    if model is None:
        cross_scores = [0.0 for _ in chunks]
    else:
        pairs = [[query, _content_without_metadata_header(chunk.content)[:3000]] for chunk in chunks]
        cross_scores = [float(item) for item in await asyncio.to_thread(model.predict, pairs)]

    cross_norm = _normalize_scores(cross_scores)
    query_tokens = set(_tokenize(query))
    intent = _query_intent(query)

    scored: list[tuple[Chunk, float]] = []
    for idx, chunk in enumerate(chunks):
        heuristic_score = _heuristic_chunk_score(query_tokens, chunk, intent)
        combined_score = heuristic_score + (cross_norm[idx] * 2.0)
        scored.append((chunk, combined_score))

    return sorted(scored, key=lambda item: item[1], reverse=True)


async def rerank_candidates(query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
    frequency = Counter(chunk.id for chunk in chunks)
    dedup: dict[int, Chunk] = {}
    for chunk in chunks:
        dedup[chunk.id] = chunk

    candidate_chunks = list(dedup.values())
    intent = _query_intent(query)
    if intent["needs_ds"]:
        ds_candidates = [
            chunk
            for chunk in candidate_chunks
            if _contains_any(_content_without_metadata_header(chunk.content).lower(), DS_TERMS)
        ]
        if ds_candidates:
            candidate_chunks = ds_candidates

    reranked = await _rerank(query, candidate_chunks)
    boosted = [
        (chunk, score + (frequency.get(chunk.id, 1) * 0.35))
        for chunk, score in reranked
    ]
    boosted.sort(key=lambda item: item[1], reverse=True)
    return [chunk for chunk, _ in boosted[:top_k]]


async def hybrid_search(
    query: str,
    db: AsyncSession,
    vector_top_k: int | None = None,
    keyword_top_k: int | None = None,
    fused_top_k: int | None = None,
    final_top_k: int | None = None,
    trace_id: str | None = None,
    attempt: int | None = None,
    query_index: int | None = None,
    do_rerank: bool = True,
    embedding_cache: dict[str, list[float]] | None = None,
) -> list[Chunk]:
    vector_limit = vector_top_k or settings.vector_top_k
    keyword_limit = keyword_top_k or settings.keyword_top_k
    fused_limit = fused_top_k or settings.fused_top_k
    final_limit = final_top_k or settings.rerank_top_k
    year, quarter = _extract_temporal_hints(query)

    query_embedding = await _embed_query(query, embedding_cache=embedding_cache)

    vector_results = await _vector_search(db, query_embedding, vector_limit, year=year, quarter=quarter)
    keyword_results = await _keyword_search(db, query, keyword_limit)

    fused = _rrf_fuse(vector_results, keyword_results, fused_limit)
    if do_rerank:
        reranked = await _rerank(query, [chunk for chunk, _ in fused])
        selected_chunks = [chunk for chunk, _ in reranked[:final_limit]]
        rerank_preview = _preview_scored(reranked, "rerank_score")
    else:
        reranked = []
        selected_chunks = [chunk for chunk, _ in fused[:final_limit]]
        rerank_preview = []

    if trace_id:
        logger.info(
            "[trace:%s] retrieval.search attempt=%s query_idx=%s vector=%s keyword=%s fused=%s reranked=%s query=%r",
            trace_id,
            attempt or 1,
            query_index or 1,
            len(vector_results),
            len(keyword_results),
            len(fused),
            len(selected_chunks),
            query,
        )
        logger.info(
            "[trace:%s] retrieval.search.filters attempt=%s query_idx=%s year=%s quarter=%s",
            trace_id,
            attempt or 1,
            query_index or 1,
            year,
            quarter,
        )
        logger.info(
            "[trace:%s] retrieval.search.top attempt=%s query_idx=%s vector_top=%s keyword_top=%s fused_top=%s rerank_top=%s",
            trace_id,
            attempt or 1,
            query_index or 1,
            _preview_scored(vector_results, "distance"),
            _preview_scored(keyword_results, "bm25_score"),
            _preview_scored(fused, "rrf_score"),
            rerank_preview,
        )

    return selected_chunks
