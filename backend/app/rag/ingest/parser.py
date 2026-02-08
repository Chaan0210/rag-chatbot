# backend/app/rag/ingest/parser.py
from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf4llm
from openai import AsyncOpenAI
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import Chunk, Document

logger = logging.getLogger(__name__)

YEAR_PATTERN = re.compile(r"(20\d{2})")
QUARTER_PATTERN = re.compile(r"([1-4]Q)", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"\S+")

TABLE_MARKDOWN_PATTERN = re.compile(r"\|.+\|\s*\n\|\s*[-:| ]+\|")
TABLE_HEADER_PATTERN = re.compile(r"^\|\s*[-:| ]+\|$", re.MULTILINE)
TABLE_LAYOUT_SPLIT_PATTERN = re.compile(r"\s{2,}|\t+")
TABLE_LAYOUT_GAP_PATTERN = re.compile(r"\S(?:\s{2,}|\t)\S")
NOTE_PATTERN = re.compile(r"^(주\)|주석|각주|\*|note\b)", re.IGNORECASE)
NUMBER_TOKEN_PATTERN = re.compile(r"^\(?[-+]?\d[\d,]*(?:\.\d+)?%?\)?$")
GENERIC_TABLE_COLUMN_PATTERN = re.compile(r"\b(?:col\d+|unnamed:?\d*)\b", re.IGNORECASE)
FINANCIAL_HINT_PATTERN = re.compile(
    r"(매출|영업이익|순이익|손익|실적|재무|revenue|operating|income|profit|margin|qoq|yoy)",
    re.IGNORECASE,
)
TABLE_WEAK_HINT_PATTERN = re.compile(
    r"(단위|증감|전년|동기|전분기|전기|비교|yoy|qoq|%)",
    re.IGNORECASE,
)
YEAR_NUMBER_PATTERN = re.compile(r"^(?:19|20)\d{2}$")
QUARTER_CONTEXT_PATTERN = re.compile(r"(?:^|[^0-9a-z])(q[1-4]|[1-4]q|[1-4]분기|quarter)(?:[^0-9a-z]|$)", re.IGNORECASE)

client = AsyncOpenAI(api_key=settings.openai_api_key)

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover - optional dependency path
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    except Exception:  # pragma: no cover
        RecursiveCharacterTextSplitter = None


@dataclass
class ChunkPayload:
    page_number: int
    chunk_index: int
    content: str
    embedding_text: str
    metadata: dict


def extract_period(filename: str) -> tuple[str | None, str | None]:
    year_match = YEAR_PATTERN.search(filename)
    quarter_match = QUARTER_PATTERN.search(filename)
    year = year_match.group(1) if year_match else None
    quarter = quarter_match.group(1).upper() if quarter_match else None
    return year, quarter


def _split_long_block(block: str, max_chars: int, overlap: int = 0) -> list[str]:
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=max(0, overlap),
            length_function=len,
        )
        pieces = [piece.strip() for piece in splitter.split_text(block) if piece.strip()]
        if pieces:
            return pieces

    lines = [line for line in block.splitlines() if line.strip()]
    parts: list[str] = []
    buf: list[str] = []
    size = 0

    for line in lines:
        line_len = len(line) + 1
        if size + line_len > max_chars and buf:
            parts.append("\n".join(buf))
            buf = []
            size = 0
        buf.append(line)
        size += line_len

    if buf:
        parts.append("\n".join(buf))

    return parts or [block[:max_chars]]


def _is_table_block(block: str) -> bool:
    return bool(TABLE_MARKDOWN_PATTERN.search(block))


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return []
    return [item.strip() for item in stripped.strip("|").split("|")]


def _table_rows(block: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or "|" not in stripped:
            continue
        if TABLE_HEADER_PATTERN.match(stripped):
            continue
        cells = _split_markdown_row(stripped)
        if len(cells) >= 2:
            rows.append(cells)
    return rows


def _is_low_quality_table_block(block: str) -> bool:
    rows = _table_rows(block)
    if len(rows) < 2:
        return False

    header_joined = " ".join(rows[0]).strip()
    if header_joined and GENERIC_TABLE_COLUMN_PATTERN.search(header_joined):
        return True

    stacked_label_rows = 0
    repeated_first_cells: Counter[str] = Counter()
    for row in rows[1:]:
        first_cell = row[0]
        br_count_first = first_cell.lower().count("<br>")
        br_count_others = max((cell.lower().count("<br>") for cell in row[1:]), default=0)
        if br_count_first >= 6 and br_count_others >= 6:
            stacked_label_rows += 1
            normalized_first = _normalize_block_text(first_cell)
            if normalized_first:
                repeated_first_cells[normalized_first] += 1

    if stacked_label_rows >= 1:
        return True
    if any(count >= 2 for count in repeated_first_cells.values()):
        return True
    return False


def _contains_low_quality_table(page_text: str) -> bool:
    blocks = [block.strip() for block in re.split(r"\n{2,}", page_text) if block.strip()]
    for block in blocks:
        if not _is_table_block(block):
            continue
        if _is_low_quality_table_block(block):
            return True
    return False


def _replace_low_quality_tables(page_text: str, fallback_tables: list[str]) -> tuple[str, int]:
    blocks = [block.strip() for block in re.split(r"\n{2,}", page_text) if block.strip()]
    kept_blocks: list[str] = []
    kept_normalized: set[str] = set()
    dropped = 0

    for block in blocks:
        if _is_table_block(block) and _is_low_quality_table_block(block):
            dropped += 1
            continue
        normalized = _normalize_block_text(block)
        if normalized:
            kept_normalized.add(normalized)
        kept_blocks.append(block)

    for table in fallback_tables:
        normalized = _normalize_block_text(table)
        if not normalized or normalized in kept_normalized:
            continue
        kept_blocks.append(table.strip())
        kept_normalized.add(normalized)

    return "\n\n".join(kept_blocks).strip(), dropped


def _is_note_block(block: str) -> bool:
    first_line = block.splitlines()[0] if block.splitlines() else block
    return bool(NOTE_PATTERN.search(first_line.strip()))


def _normalize_block_text(text: str) -> str:
    return " ".join(text.split())


def _is_temporal_numeric_token(token: str, prev_token: str = "", next_token: str = "") -> bool:
    normalized = token.strip().strip("[](){}<>")
    normalized = normalized.replace(",", "")
    normalized = normalized.rstrip("%")
    normalized = normalized.lstrip("+-")
    if not normalized.isdigit():
        return False

    if YEAR_NUMBER_PATTERN.match(normalized):
        return True

    if normalized in {"1", "2", "3", "4"}:
        context = f"{prev_token} {next_token}".lower()
        if QUARTER_CONTEXT_PATTERN.search(context):
            return True

    return False


def _numeric_density(text: str) -> float:
    tokens = TOKEN_PATTERN.findall(text)
    if not tokens:
        return 0.0

    numeric_tokens = 0
    for idx, token in enumerate(tokens):
        cleaned = token.strip().strip("[](){}<>")
        if NUMBER_TOKEN_PATTERN.match(cleaned):
            prev_token = tokens[idx - 1] if idx > 0 else ""
            next_token = tokens[idx + 1] if idx + 1 < len(tokens) else ""
            if _is_temporal_numeric_token(cleaned, prev_token=prev_token, next_token=next_token):
                continue
            numeric_tokens += 1
    return numeric_tokens / max(1, len(tokens))


def _has_table_layout_signal(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False

    pipe_rows = 0
    aligned_rows = 0
    numeric_rich_rows = 0
    aligned_column_hist: Counter[int] = Counter()

    for line in lines:
        if line.count("|") >= 2:
            pipe_rows += 1

        if TABLE_LAYOUT_GAP_PATTERN.search(line):
            aligned_rows += 1
            columns = [item for item in TABLE_LAYOUT_SPLIT_PATTERN.split(line) if item.strip()]
            if len(columns) >= 3:
                aligned_column_hist[len(columns)] += 1

        line_numeric_tokens = 0
        for token in TOKEN_PATTERN.findall(line):
            cleaned = token.strip().strip("[](){}<>")
            if NUMBER_TOKEN_PATTERN.match(cleaned):
                line_numeric_tokens += 1
        if line_numeric_tokens >= 3:
            numeric_rich_rows += 1

    if pipe_rows >= 2:
        return True
    if aligned_rows >= 3 and any(count >= 2 for count in aligned_column_hist.values()):
        return True
    if aligned_rows >= 2 and numeric_rich_rows >= 3:
        return True
    return False


def _should_attempt_table_fallback(page_text: str) -> bool:
    body = page_text.strip()
    if not body:
        return True

    # Already has markdown table structure.
    if _is_table_block(body) or TABLE_HEADER_PATTERN.search(body):
        if _contains_low_quality_table(body):
            return True
        return False

    density = _numeric_density(body)
    has_financial_hint = bool(FINANCIAL_HINT_PATTERN.search(body))
    length = len(body)
    has_layout_signal = _has_table_layout_signal(body)
    has_weak_table_signal = bool(TABLE_WEAK_HINT_PATTERN.search(body))

    # Long, mostly narrative pages should not trigger expensive table fallback.
    if length >= 900 and density < 0.08 and not has_layout_signal:
        return False

    # Weak table hints should be supported by at least one structural signal.
    if has_financial_hint and has_weak_table_signal and (has_layout_signal or density >= 0.05):
        return True

    # Very short extraction with some numbers often means table content was missed.
    if length < 240 and density >= 0.05:
        return True

    # Financial pages still prefer table-layout hints for precision.
    if has_financial_hint and has_layout_signal and density >= 0.05:
        return True

    if has_layout_signal and density >= 0.05:
        return True

    if length < 420 and has_financial_hint and density >= 0.14:
        return True

    return False


def _table_head_signals(
    table_blocks: list[str],
    max_tables: int = 2,
    max_lines_per_table: int = 2,
    max_chars: int = 220,
) -> list[str]:
    signals: list[str] = []
    for block in table_blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        selected_lines: list[str] = []
        for line in lines:
            if "|" not in line:
                continue
            if TABLE_HEADER_PATTERN.match(line):
                continue
            normalized = _normalize_block_text(line.replace("|", " "))
            if not normalized:
                continue
            selected_lines.append(normalized)
            if len(selected_lines) >= max_lines_per_table:
                break

        if not selected_lines:
            fallback = _normalize_block_text(lines[0])
            if fallback:
                selected_lines.append(fallback)

        if not selected_lines:
            continue

        signal = f"[TABLE_HEAD] {' / '.join(selected_lines)}"
        signals.append(signal[:max_chars].strip())
        if len(signals) >= max_tables:
            break

    return signals


def _note_head_signals(note_blocks: list[str], max_notes: int = 2, max_chars: int = 160) -> list[str]:
    signals: list[str] = []
    for block in note_blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        first_line = _normalize_block_text(lines[0])
        if not first_line:
            continue
        signals.append(f"[NOTE_HEAD] {first_line[:max_chars].strip()}")
        if len(signals) >= max_notes:
            break
    return signals


def _make_chunk_bodies(
    page_text: str,
    max_chars: int,
    overlap: int,
    dedup_scope: str = "",
) -> list[tuple[str, str]]:
    blocks = [block.strip() for block in re.split(r"\n{2,}", page_text) if block.strip()]
    if not blocks:
        return []

    table_blocks: list[str] = []
    note_blocks: list[str] = []
    body_blocks: list[str] = []

    for block in blocks:
        if _is_table_block(block):
            table_blocks.append(block)
            continue
        if _is_note_block(block):
            note_blocks.append(block)
            continue
        body_blocks.append(block)

    # Slide chunks use body-only text to avoid duplicating table/note content.
    chunk_pairs: list[tuple[str, str]] = []
    body_text = "\n\n".join(body_blocks).strip()
    signal_lines = [*_table_head_signals(table_blocks), *_note_head_signals(note_blocks)]
    if signal_lines:
        signal_blob = "\n".join(signal_lines).strip()
        body_text = f"{body_text}\n\n{signal_blob}".strip() if body_text else signal_blob
    if body_text:
        slide_chunks = _split_long_block(body_text, max_chars=max_chars, overlap=overlap)
        chunk_pairs.extend((chunk, "slide") for chunk in slide_chunks)

    # Table / note chunks are indexed separately for retrieval precision.
    for block in table_blocks:
        for piece in _split_long_block(block, max_chars=max_chars, overlap=0):
            chunk_pairs.append((piece, "table"))
    for block in note_blocks:
        for piece in _split_long_block(block, max_chars=max_chars, overlap=0):
            chunk_pairs.append((piece, "note"))

    dedup: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for body, chunk_type in chunk_pairs:
        key = (dedup_scope, chunk_type, " ".join(body.split()))
        if key in seen:
            continue
        seen.add(key)
        dedup.append((body, chunk_type))

    return dedup


async def _embed_texts(texts: list[str]) -> list[list[float]]:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for ingestion embedding generation.")

    vectors: list[list[float]] = []
    batch_size = max(1, settings.embedding_batch_size)

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = await client.embeddings.create(model=settings.embedding_model, input=batch)
        vectors.extend([_normalize_vector(item.embedding) for item in response.data])

    return vectors


def _embedding_input(text: str) -> str:
    limit = max(1, settings.embedding_max_chars)
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    logger.warning("Embedding input truncated from %s to %s chars", len(stripped), limit)
    return stripped[:limit].strip()


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        return vector
    return [value / norm for value in vector]


def _df_to_markdown(df: Any) -> str:
    columns = [str(col) for col in getattr(df, "columns", [])]
    rows = []
    for row in getattr(df, "values", []):
        rows.append([str(item) for item in row])

    if not columns and rows:
        columns = [f"col_{idx + 1}" for idx in range(len(rows[0]))]

    if not columns:
        return ""

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---" for _ in columns]) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body]).strip()


def _extract_tables_with_camelot(file_path: str, page_number: int) -> tuple[list[str], str | None]:
    try:
        import camelot  # type: ignore
    except Exception as exc:
        logger.warning("Camelot unavailable, disable fallback for this process: %s", exc)
        return [], "unavailable"

    try:
        tables = camelot.read_pdf(file_path, pages=str(page_number), flavor="stream")
    except Exception as exc:
        logger.warning("Camelot read failed at page %s: %s", page_number, exc)
        return [], "page_error"

    markdown_tables: list[str] = []
    for table in tables:
        markdown = _df_to_markdown(table.df)
        if markdown:
            markdown_tables.append(markdown)
    return markdown_tables, None


def _extract_tables_with_tabula(file_path: str, page_number: int) -> tuple[list[str], str | None]:
    try:
        import tabula  # type: ignore
    except Exception as exc:
        logger.warning("Tabula unavailable, disable fallback for this process: %s", exc)
        return [], "unavailable"

    table_frames: list[Any] = []
    errors: list[str] = []
    succeeded_without_exception = False
    for page_spec in (str(page_number), page_number):
        try:
            tables = tabula.read_pdf(file_path, pages=page_spec, multiple_tables=True)
            succeeded_without_exception = True
            if tables:
                table_frames = list(tables)
                break
        except Exception as exc:
            errors.append(f"pages={page_spec!r}: {exc}")

    if not table_frames and errors and not succeeded_without_exception:
        logger.warning("Tabula read failed at page %s: %s", page_number, " | ".join(errors))
        return [], "page_error"

    markdown_tables: list[str] = []
    for table in table_frames:
        markdown = _df_to_markdown(table)
        if markdown:
            markdown_tables.append(markdown)
    return markdown_tables, None


async def _extract_tables_fallback(
    file_path: str,
    page_number: int,
    tool_state: dict[str, Any] | None = None,
) -> list[str]:
    state = tool_state if tool_state is not None else {}
    state["fallback_calls"] = int(state.get("fallback_calls", 0)) + 1
    fallback_pages_attempted = state.setdefault("fallback_pages_attempted", set())
    fallback_pages_with_tables = state.setdefault("fallback_pages_with_tables", set())
    if isinstance(fallback_pages_attempted, set):
        fallback_pages_attempted.add(page_number)

    camelot_unavailable = bool(state.get("camelot_unavailable", False))
    tabula_unavailable = bool(state.get("tabula_unavailable", False))
    camelot_failed_pages = state.setdefault("camelot_failed_pages", set())
    tabula_failed_pages = state.setdefault("tabula_failed_pages", set())

    if not isinstance(camelot_failed_pages, set):
        camelot_failed_pages = set()
        state["camelot_failed_pages"] = camelot_failed_pages
    if not isinstance(tabula_failed_pages, set):
        tabula_failed_pages = set()
        state["tabula_failed_pages"] = tabula_failed_pages

    if not camelot_unavailable and page_number not in camelot_failed_pages:
        state["camelot_calls"] = int(state.get("camelot_calls", 0)) + 1
        camelot_tables, camelot_status = await asyncio.to_thread(_extract_tables_with_camelot, file_path, page_number)
        if camelot_status == "unavailable":
            state["camelot_unavailable"] = True
        elif camelot_status == "page_error":
            camelot_failed_pages.add(page_number)
            state["camelot_page_errors"] = int(state.get("camelot_page_errors", 0)) + 1
        if camelot_tables:
            if isinstance(fallback_pages_with_tables, set):
                fallback_pages_with_tables.add(page_number)
            state["camelot_hits"] = int(state.get("camelot_hits", 0)) + 1
            return camelot_tables

    if not tabula_unavailable and page_number not in tabula_failed_pages:
        state["tabula_calls"] = int(state.get("tabula_calls", 0)) + 1
        tabula_tables, tabula_status = await asyncio.to_thread(_extract_tables_with_tabula, file_path, page_number)
        if tabula_status == "unavailable":
            state["tabula_unavailable"] = True
        elif tabula_status == "page_error":
            tabula_failed_pages.add(page_number)
            state["tabula_page_errors"] = int(state.get("tabula_page_errors", 0)) + 1
        if tabula_tables:
            if isinstance(fallback_pages_with_tables, set):
                fallback_pages_with_tables.add(page_number)
            state["tabula_hits"] = int(state.get("tabula_hits", 0)) + 1
            return tabula_tables

    return []


def _extract_pages_with_pymupdf(file_path: str) -> list[dict]:
    try:
        import fitz  # type: ignore
    except Exception:
        return []

    pages: list[dict] = []
    with fitz.open(file_path) as doc:
        for page_idx, page in enumerate(doc, start=1):
            pages.append({"text": page.get_text("text"), "metadata": {"page": page_idx}})

    return pages


async def _load_markdown_pages(file_path: str) -> list[dict]:
    try:
        markdown_pages = await asyncio.to_thread(
            pymupdf4llm.to_markdown,
            file_path,
            page_chunks=True,
        )
    except Exception as exc:
        logger.warning("pymupdf4llm parsing failed, fallback to PyMuPDF text extraction: %s", exc)
        markdown_pages = []

    if isinstance(markdown_pages, str):
        return [{"text": markdown_pages, "metadata": {"page": 1}}]

    if isinstance(markdown_pages, list) and markdown_pages:
        return markdown_pages

    return await asyncio.to_thread(_extract_pages_with_pymupdf, file_path)


async def process_pdf(file_path: str, filename: str, db: AsyncSession) -> dict:
    logger.info("Start ingestion: %s", filename)
    year, quarter = extract_period(filename)

    markdown_pages = await _load_markdown_pages(file_path)
    raw_pages: list[dict] | None = None
    table_fallback_state: dict[str, Any] = {}
    try:
        existing_document = await db.scalar(select(Document).where(Document.filename == filename))
        if existing_document is not None:
            # Explicit chunk cleanup protects against environments where FK cascade
            # was not applied as expected.
            await db.execute(delete(Chunk).where(Chunk.document_id == existing_document.id))
            await db.delete(existing_document)
            await db.flush()

        document = Document(filename=filename, year=year, quarter=quarter)
        db.add(document)
        await db.flush()

        chunk_payloads: list[ChunkPayload] = []

        for fallback_page_number, page in enumerate(markdown_pages, start=1):
            page_metadata = page.get("metadata", {}) if isinstance(page, dict) else {}
            page_number_raw = page_metadata.get("page") if isinstance(page_metadata, dict) else None
            try:
                parsed_page_number = int(page_number_raw or fallback_page_number)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid page metadata value %r in %s. fallback to page=%s",
                    page_number_raw,
                    filename,
                    fallback_page_number,
                )
                parsed_page_number = fallback_page_number
            if parsed_page_number < 1:
                logger.warning(
                    "Out-of-range page metadata %r in %s. fallback to page=%s",
                    parsed_page_number,
                    filename,
                    fallback_page_number,
                )
                parsed_page_number = fallback_page_number

            metadata_page_number = parsed_page_number
            effective_page_number = fallback_page_number
            if metadata_page_number != fallback_page_number:
                logger.info(
                    "Page metadata mismatch in %s: metadata_page=%s enumerate_page=%s; using enumerate page for storage and fallback IO",
                    filename,
                    metadata_page_number,
                    fallback_page_number,
                )
            page_text = (page.get("text", "") if isinstance(page, dict) else "").strip()

            fallback_raw = ""
            if not page_text:
                if raw_pages is None:
                    raw_pages = await asyncio.to_thread(_extract_pages_with_pymupdf, file_path)
                if raw_pages:
                    primary_index = fallback_page_number - 1
                    secondary_index = metadata_page_number - 1
                    if 0 <= primary_index < len(raw_pages):
                        fallback_raw = str(raw_pages[primary_index].get("text", "")).strip()
                    elif 0 <= secondary_index < len(raw_pages):
                        fallback_raw = str(raw_pages[secondary_index].get("text", "")).strip()
                    else:
                        logger.warning(
                            "fallback/enumerate page out of raw range for %s: metadata_page=%s enumerate_page=%s raw_pages=%s",
                            filename,
                            metadata_page_number,
                            fallback_page_number,
                            len(raw_pages),
                        )
            fallback_tables: list[str] = []
            if _should_attempt_table_fallback(page_text):
                fallback_tables = await _extract_tables_fallback(
                    file_path,
                    effective_page_number,
                    tool_state=table_fallback_state,
                )
                if not fallback_tables and metadata_page_number != effective_page_number:
                    fallback_tables = await _extract_tables_fallback(
                        file_path,
                        metadata_page_number,
                        tool_state=table_fallback_state,
                    )

            if not page_text:
                page_text = "\n\n".join([fallback_raw, *fallback_tables]).strip()
            elif fallback_tables:
                if _contains_low_quality_table(page_text):
                    replaced_page_text, dropped_blocks = _replace_low_quality_tables(page_text, fallback_tables)
                    if replaced_page_text:
                        logger.info(
                            "Replaced low-quality table blocks with fallback tables file=%s page=%s dropped_blocks=%s fallback_tables=%s",
                            filename,
                            effective_page_number,
                            dropped_blocks,
                            len(fallback_tables),
                        )
                        page_text = replaced_page_text
                else:
                    existing_blocks = {
                        _normalize_block_text(block)
                        for block in re.split(r"\n{2,}", page_text)
                        if block.strip()
                    }
                    new_tables = [
                        table
                        for table in fallback_tables
                        if table.strip() and _normalize_block_text(table) not in existing_blocks
                    ]
                    if new_tables:
                        page_text = "\n\n".join([page_text, *new_tables]).strip()

            if not page_text:
                continue

            page_chunks = _make_chunk_bodies(
                page_text,
                max_chars=settings.chunk_size,
                overlap=settings.chunk_overlap,
                dedup_scope=f"{filename}:{effective_page_number}",
            )

            for chunk_index, (body, chunk_type) in enumerate(page_chunks):
                metadata = {
                    "source": filename,
                    "year": year,
                    "quarter": quarter,
                    "page": effective_page_number,
                    "chunk_index": chunk_index,
                    "chunk_type": chunk_type,
                }
                if metadata_page_number != effective_page_number:
                    metadata["source_page_metadata"] = metadata_page_number
                cleaned_body = body.strip()
                chunk_payloads.append(
                    ChunkPayload(
                        page_number=effective_page_number,
                        chunk_index=chunk_index,
                        content=cleaned_body,
                        embedding_text=cleaned_body,
                        metadata=metadata,
                    )
                )

        fallback_calls = int(table_fallback_state.get("fallback_calls", 0))
        if fallback_calls > 0:
            pages_attempted = table_fallback_state.get("fallback_pages_attempted", set())
            pages_with_tables = table_fallback_state.get("fallback_pages_with_tables", set())
            attempted_count = len(pages_attempted) if isinstance(pages_attempted, set) else 0
            success_count = len(pages_with_tables) if isinstance(pages_with_tables, set) else 0
            logger.info(
                "table_fallback summary file=%s calls=%s pages_attempted=%s pages_with_tables=%s camelot_calls=%s camelot_hits=%s camelot_page_errors=%s tabula_calls=%s tabula_hits=%s tabula_page_errors=%s camelot_unavailable=%s tabula_unavailable=%s",
                filename,
                fallback_calls,
                attempted_count,
                success_count,
                int(table_fallback_state.get("camelot_calls", 0)),
                int(table_fallback_state.get("camelot_hits", 0)),
                int(table_fallback_state.get("camelot_page_errors", 0)),
                int(table_fallback_state.get("tabula_calls", 0)),
                int(table_fallback_state.get("tabula_hits", 0)),
                int(table_fallback_state.get("tabula_page_errors", 0)),
                bool(table_fallback_state.get("camelot_unavailable", False)),
                bool(table_fallback_state.get("tabula_unavailable", False)),
            )

        if not chunk_payloads:
            await db.commit()
            return {
                "filename": filename,
                "document_id": document.id,
                "year": year,
                "quarter": quarter,
                "pages": len(markdown_pages),
                "chunks": 0,
                "saved_path": str(Path(file_path)),
            }

        embedding_inputs = [_embedding_input(chunk.embedding_text) for chunk in chunk_payloads]
        embeddings = await _embed_texts(embedding_inputs)
        if len(embeddings) != len(chunk_payloads):
            raise RuntimeError(
                "Embedding count mismatch during ingestion "
                f"(chunks={len(chunk_payloads)}, embeddings={len(embeddings)}) for {filename}"
            )

        db.add_all(
            [
                Chunk(
                    document_id=document.id,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    metadata_json=chunk.metadata,
                    embedding=vector,
                )
                for chunk, vector in zip(chunk_payloads, embeddings, strict=True)
            ]
        )

        await db.commit()

        token_count = sum(len(TOKEN_PATTERN.findall(chunk.embedding_text)) for chunk in chunk_payloads)
        logger.info("Ingestion completed: %s (%s chunks)", filename, len(chunk_payloads))

        return {
            "filename": filename,
            "document_id": document.id,
            "year": year,
            "quarter": quarter,
            "pages": len(markdown_pages),
            "chunks": len(chunk_payloads),
            "token_estimate": token_count,
            "saved_path": str(Path(file_path)),
        }
    except Exception:
        await db.rollback()
        logger.exception("process_pdf failed and rolled back: %s", filename)
        raise


async def ingest_existing_pdfs(pdf_dir: str, db: AsyncSession, force_reindex: bool = False) -> dict:
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        return {"total_files": 0, "ingested": [], "skipped": [], "failed": []}

    files = sorted([item for item in pdf_path.glob("*.pdf") if item.is_file()])
    ingested: list[str] = []
    skipped: list[str] = []
    failed: list[dict[str, str]] = []

    for file in files:
        filename = file.name
        exists = await db.scalar(select(Document.id).where(Document.filename == filename))
        if exists is not None and not force_reindex:
            skipped.append(filename)
            continue

        try:
            await process_pdf(str(file), filename, db)
            ingested.append(filename)
        except Exception as exc:
            await db.rollback()
            logger.exception("Failed to ingest existing PDF: %s", filename)
            failed.append({"filename": filename, "error": str(exc)})

    return {
        "total_files": len(files),
        "ingested": ingested,
        "skipped": skipped,
        "failed": failed,
    }
