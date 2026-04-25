"""Central configuration loader for RAG-Tag Force.

Loads environment variables from .env and provides typed, path-safe
access to all configuration values used across the application.
"""

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import os

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _get_path(env_var: str, default: str) -> Path:
    """Resolve an environment variable to an absolute Path."""
    raw = os.getenv(env_var, default)
    path = Path(raw)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


# ── Paths ────────────────────────────────────────────────────────────
CHROMA_PATH: Path = _get_path("CHROMA_PATH", "./data/chroma")
SQLITE_PATH: Path = _get_path("SQLITE_PATH", "./data/structured/entitlements.db")
ONTOLOGY_PATH: Path = _get_path("ONTOLOGY_PATH", "./ontology/military_entitlements.ttl")
DATA_RAW_PATH: Path = _PROJECT_ROOT / "data" / "raw"
DATA_STRUCTURED_PATH: Path = _PROJECT_ROOT / "data" / "structured"

# ── LLM ──────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-20250506")

# ── Embeddings ───────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"

# ── Retrieval defaults ───────────────────────────────────────────────
DEFAULT_TOP_K: int = 5

# ── Project root (for scripts) ───────────────────────────────────────
PROJECT_ROOT: Path = _PROJECT_ROOT
