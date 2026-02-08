from collections.abc import AsyncIterator
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.db.base import Base

logger = logging.getLogger(__name__)

engine = create_async_engine(settings.database_url, future=True, echo=False)
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    # Ensure model modules are imported before create_all so tables are registered.
    from app.db import models as _models  # noqa: F401

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_chunks_content_trgm ON chunks USING GIN (content gin_trgm_ops)")
        )
        # pgvector ivfflat index currently supports up to 2000-d vectors.
        if settings.embedding_dimension <= 2000:
            try:
                await conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_cosine "
                        "ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
                    )
                )
            except Exception as exc:  # pragma: no cover - index strategy may vary by environment
                logger.warning("Skipping cosine ivfflat index creation: %s", exc)
        else:
            logger.info(
                "Skipping ivfflat cosine index because EMBEDDING_DIMENSION=%s exceeds pgvector ivfflat limit(2000).",
                settings.embedding_dimension,
            )
