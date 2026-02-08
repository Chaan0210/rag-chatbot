from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.chat import UploadResponse
from app.core.config import settings
from app.core.paths import resolve_pdf_dir
from app.db.session import get_db
from app.rag.ingest.parser import process_pdf

logger = logging.getLogger(__name__)
router = APIRouter(prefix=settings.api_prefix, tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)) -> UploadResponse:
    filename = file.filename or "uploaded.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    pdf_dir = resolve_pdf_dir(settings.data_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    file_path = pdf_dir / filename
    file_bytes = await file.read()
    file_path.write_bytes(file_bytes)

    try:
        result = await process_pdf(str(file_path), filename, db)
        return UploadResponse(**result)
    except Exception as exc:
        await db.rollback()
        logger.exception("Upload ingestion failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to process PDF") from exc
