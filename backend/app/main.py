from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.chat import router as chat_router
from app.api.routes.sessions import router as sessions_router
from app.api.routes.upload import router as upload_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.paths import resolve_pdf_dir
from app.db.session import AsyncSessionLocal, init_db
from app.rag.ingest.parser import ingest_existing_pdfs

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    if settings.auto_ingest_on_startup:
        pdf_dir = resolve_pdf_dir(settings.data_dir)
        logger.info("Startup PDF directory: %s", pdf_dir)
        async with AsyncSessionLocal() as session:
            summary = await ingest_existing_pdfs(str(pdf_dir), session, force_reindex=False)
            logger.info("Startup ingestion summary: %s", summary)
    yield


app = FastAPI(title=settings.project_name, lifespan=lifespan)

if settings.cors_allow_origins.strip() == "*":
    origins = ["*"]
else:
    origins = [origin.strip() for origin in settings.cors_allow_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


app.include_router(upload_router)
app.include_router(chat_router)
app.include_router(sessions_router)
