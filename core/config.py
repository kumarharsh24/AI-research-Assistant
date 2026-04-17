from __future__ import annotations

import logging
import os
from dataclasses import dataclass


logger = logging.getLogger(__name__)


_PLACEHOLDER_VALUES = {
    "YOUR_API_KEY",
    "YOUR_GEMINI_API_KEY",
    "YOUR_TAVILY_API_KEY",
}


@dataclass
class Settings:
    gemini_api_key: str
    tavily_api_key: str
    gemini_model: str
    index_dir: str
    default_top_k: int


def is_secret_configured(value: str) -> bool:
    cleaned = value.strip()
    if not cleaned:
        return False
    if cleaned in _PLACEHOLDER_VALUES:
        return False
    if cleaned.upper().startswith("YOUR_"):
        return False
    return True


def _read_streamlit_secret(name: str) -> str:
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        return ""
    return ""


def _read_setting(name: str, default: str = "") -> str:
    from_secret = _read_streamlit_secret(name)
    if from_secret:
        return from_secret

    from_env = os.getenv(name, "").strip()
    if from_env:
        return from_env

    return default


def _safe_int_env(name: str, default: int, min_value: int = 1, max_value: int = 20) -> int:
    raw_value = _read_setting(name, str(default)).strip()
    try:
        parsed = int(raw_value)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw_value, default)
        return default

    if parsed < min_value or parsed > max_value:
        logger.warning(
            "%s=%s out of range [%s, %s]; using default %s",
            name,
            parsed,
            min_value,
            max_value,
            default,
        )
        return default
    return parsed


def get_settings() -> Settings:
    """Read runtime settings from environment variables."""
    return Settings(
        gemini_api_key=_read_setting("GEMINI_API_KEY", ""),
        tavily_api_key=_read_setting("TAVILY_API_KEY", ""),
        gemini_model=_read_setting("GEMINI_MODEL", "gemini-2.0-flash"),
        index_dir=_read_setting("FAISS_INDEX_DIR", "data/faiss_index"),
        default_top_k=_safe_int_env("RAG_TOP_K", default=4, min_value=1, max_value=20),
    )
