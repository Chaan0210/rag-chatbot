from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    # backend/app/core/paths.py -> backend/app/core -> backend/app -> backend -> project root
    return Path(__file__).resolve().parents[3]


def resolve_data_dir(data_dir: str) -> Path:
    raw = Path(data_dir)
    if raw.is_absolute():
        return raw

    root_candidate = project_root() / raw
    cwd_candidate = Path.cwd() / raw

    # Prefer repository-root relative path for stable behavior regardless of run cwd.
    if root_candidate.exists() or not cwd_candidate.exists():
        return root_candidate

    return cwd_candidate


def resolve_pdf_dir(data_dir: str) -> Path:
    primary = resolve_data_dir(data_dir) / "pdfs"
    root_default = project_root() / "data" / "pdfs"

    primary_has_pdf = primary.exists() and any(primary.glob("*.pdf"))
    root_has_pdf = root_default.exists() and any(root_default.glob("*.pdf"))

    if primary_has_pdf:
        return primary

    if root_has_pdf and root_default != primary:
        return root_default

    return primary
