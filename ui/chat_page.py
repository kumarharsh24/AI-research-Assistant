from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent.agent import build_agent_executor, run_agent_turn
from core.config import get_settings, is_secret_configured
from rag.ingest import ingest_pdf
from rag.retriever import load_retriever

logger = logging.getLogger(__name__)


PROJECT_NAME = "AI Research Assistant"
PROJECT_TAGLINE = "RAG, Memory, and Live Web Intelligence"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

            :root {
                --bg-ink: #060e1d;
                --bg-mid: #0c1a30;
                --panel: rgba(10, 24, 46, 0.82);
                --panel-solid: #0d1f3a;
                --line: #2b4367;
                --text-main: #eef4ff;
                --text-soft: #b9cae4;
                --accent-warm: #ff8d42;
                --accent-cool: #21d0b3;
            }

            .stApp {
                background:
                    radial-gradient(1200px 700px at 86% -12%, rgba(255, 141, 66, 0.20), transparent 56%),
                    radial-gradient(900px 650px at -8% 16%, rgba(33, 208, 179, 0.19), transparent 58%),
                    linear-gradient(150deg, var(--bg-mid), var(--bg-ink) 60%);
                color: var(--text-main);
                font-family: "IBM Plex Sans", sans-serif;
            }

            .block-container {
                padding-top: 2.2rem;
                max-width: 1200px;
            }

            h1, h2, h3, h4 {
                font-family: "Space Grotesk", sans-serif;
                letter-spacing: 0.2px;
            }

            .hero-card {
                border: 1px solid var(--line);
                border-radius: 20px;
                background: linear-gradient(140deg, rgba(11, 27, 51, 0.92), rgba(9, 20, 39, 0.82));
                padding: 1.25rem 1.35rem 1.15rem;
                margin: 0.15rem 0 1.2rem;
                box-shadow: 0 24px 46px rgba(0, 0, 0, 0.32);
            }

            .hero-kicker {
                margin: 0 0 0.35rem;
                color: var(--accent-cool);
                text-transform: uppercase;
                letter-spacing: 0.13em;
                font-size: 0.76rem;
                font-weight: 600;
            }

            .hero-title {
                margin: 0;
                font-size: clamp(1.85rem, 3.5vw, 2.65rem);
                line-height: 1.1;
                color: var(--text-main);
            }

            .hero-subtitle {
                margin: 0.55rem 0 0;
                color: var(--text-soft);
                max-width: 74ch;
                font-size: 0.99rem;
            }

            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(23, 30, 49, 0.96), rgba(9, 17, 32, 0.98));
                border-right: 1px solid var(--line);
            }

            section[data-testid="stSidebar"] * {
                color: #edf2fc;
            }

            div[data-testid="stTextInput"] input {
                background: rgba(7, 16, 31, 0.94);
                border: 1px solid var(--line);
                border-radius: 12px;
                color: #f3f7ff;
            }

            div[data-testid="stTextInput"] input:focus {
                border-color: var(--accent-cool);
                box-shadow: 0 0 0 1px var(--accent-cool);
            }

            div[data-testid="stFileUploader"] section {
                background: rgba(8, 18, 34, 0.82);
                border: 1px dashed var(--line);
                border-radius: 14px;
            }

            div.stButton > button {
                width: 100%;
                border-radius: 12px;
                border: 1px solid rgba(255, 141, 66, 0.62);
                background: linear-gradient(120deg, rgba(255, 141, 66, 0.90), rgba(255, 122, 83, 0.82));
                color: #081325;
                font-weight: 700;
                transition: transform 0.15s ease, box-shadow 0.2s ease;
            }

            div.stButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 12px 24px rgba(255, 141, 66, 0.30);
            }

            [data-testid="stChatMessage"] {
                border: 1px solid rgba(45, 67, 102, 0.9);
                border-radius: 14px;
                background: rgba(8, 20, 39, 0.86);
                padding: 0.22rem 0.72rem;
                margin-bottom: 0.75rem;
            }

            [data-testid="stChatInput"] {
                border: 1px solid var(--line);
                border-radius: 14px;
                background: rgba(10, 21, 40, 0.94);
            }

            .status-strip {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
                gap: 0.6rem;
                margin: 0.2rem 0 1rem;
            }

            .status-chip {
                border: 1px solid var(--line);
                border-radius: 14px;
                background: rgba(8, 20, 39, 0.86);
                padding: 0.58rem 0.76rem;
            }

            .status-label {
                font-size: 0.72rem;
                text-transform: uppercase;
                letter-spacing: 0.09em;
                color: var(--text-soft);
                margin-bottom: 0.16rem;
                font-weight: 600;
            }

            .status-value {
                font-size: 0.95rem;
                font-weight: 700;
                color: var(--text-main);
            }

            .status-ok {
                box-shadow: inset 0 0 0 1px rgba(33, 208, 179, 0.35);
            }

            .status-warn {
                box-shadow: inset 0 0 0 1px rgba(255, 141, 66, 0.40);
            }

            .status-info {
                box-shadow: inset 0 0 0 1px rgba(125, 175, 255, 0.35);
            }

            button[aria-label="Show password text"],
            button[aria-label="Hide password text"] {
                display: none !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <p class="hero-kicker">Adaptive research cockpit</p>
            <h1 class="hero-title">{PROJECT_NAME}</h1>
            <p class="hero-subtitle">
                {PROJECT_TAGLINE}. Blend conversational memory, document intelligence,
                and live web context in one focused workspace.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _clear_faiss_index(index_dir: str) -> None:
    index_path = Path(index_dir)
    if not index_path.exists():
        return

    for file_path in index_path.glob("*"):
        if file_path.is_file():
            file_path.unlink(missing_ok=True)
    index_path.rmdir()


def _sanitize_runtime_error(error_text: str) -> str:
    text = error_text.strip()
    if not text:
        return "Unknown runtime error."

    lowered = text.lower()
    if "resource_exhausted" in lowered or "quota" in lowered:
        return (
            "Gemini API quota is exhausted for this key/project. "
            "Check quota/billing, wait for reset, or use another API key."
        )
    if "not_found" in lowered and "models/" in lowered:
        return (
            "Configured Gemini model is unavailable for this API key. "
            "Set GEMINI_MODEL to a supported model, e.g. gemini-2.0-flash."
        )
    if "api key" in lowered:
        return "Server-side API key configuration is missing or invalid."
    if len(text) > 220:
        return text[:217] + "..."
    return text


def _ensure_agent(gemini_api_key: str, model_name: str) -> None:
    if not is_secret_configured(gemini_api_key):
        st.session_state.agent_executor = None
        st.session_state.agent_error = "GEMINI_API_KEY is missing in server-side secrets (.env or Streamlit Secrets)."
        return

    if st.session_state.get("agent_executor") is not None:
        return

    try:
        st.session_state.agent_executor = build_agent_executor(
            gemini_api_key=gemini_api_key,
            model_name=model_name,
            retriever=st.session_state.get("retriever"),
        )
        st.session_state.agent_error = None
    except Exception as exc:
        logger.exception("Auto agent initialization failed")
        st.session_state.agent_executor = None
        st.session_state.agent_error = _sanitize_runtime_error(str(exc))


def _render_runtime_status(
    gemini_ready: bool,
    tavily_ready: bool,
    model_name: str,
    retriever_ready: bool,
    agent_ready: bool,
    agent_error: str | None,
) -> None:
    def chip(label: str, value: str, tone: str) -> str:
        return (
            f'<div class="status-chip {tone}">'
            f'<div class="status-label">{label}</div>'
            f'<div class="status-value">{value}</div>'
            "</div>"
        )

    markup = "".join(
        [
            chip("Gemini", "Configured" if gemini_ready else "Missing", "status-ok" if gemini_ready else "status-warn"),
            chip("Web Search", "Enabled" if tavily_ready else "Disabled", "status-ok" if tavily_ready else "status-info"),
            chip("Model", model_name, "status-info"),
            chip("Document Index", "Ready" if retriever_ready else "Not Loaded", "status-ok" if retriever_ready else "status-warn"),
            chip("Agent", "Ready" if agent_ready else "Unavailable", "status-ok" if agent_ready else "status-warn"),
        ]
    )
    st.markdown(f'<div class="status-strip">{markup}</div>', unsafe_allow_html=True)

    if agent_error:
        trimmed_error = agent_error.strip()
        if len(trimmed_error) > 220:
            trimmed_error = trimmed_error[:217] + "..."
        st.warning(f"Runtime notice: {trimmed_error}")


def render_app() -> None:
    load_dotenv()

    st.set_page_config(page_title=PROJECT_NAME, page_icon="A", layout="wide")
    _inject_styles()
    _render_hero()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = None
    if "agent_error" not in st.session_state:
        st.session_state.agent_error = None

    settings = get_settings()
    gemini_api_key = settings.gemini_api_key.strip()
    tavily_api_key = settings.tavily_api_key.strip()
    model_name = settings.gemini_model.strip() or "gemini-2.0-flash"
    gemini_ready = is_secret_configured(gemini_api_key)
    tavily_ready = is_secret_configured(tavily_api_key)

    with st.sidebar:
        st.header("Document RAG")
        uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

        col1, col2 = st.columns(2)
        process_clicked = col1.button("Process PDF")
        clear_index_clicked = col2.button("Clear Index")

        if clear_index_clicked:
            try:
                _clear_faiss_index(settings.index_dir)
                st.session_state.retriever = None
                st.session_state.agent_executor = None
                st.success("FAISS index cleared.")
            except Exception as exc:
                logger.exception("Failed to clear FAISS index")
                st.error(f"Could not clear index: {exc}")

        if process_clicked:
            if not gemini_ready:
                st.error("GEMINI_API_KEY is missing in server-side secrets (.env or Streamlit Secrets).")
            elif uploaded_pdf is None:
                st.error("Please upload a PDF first.")
            else:
                os.environ["GEMINI_API_KEY"] = gemini_api_key
                if tavily_ready:
                    os.environ["TAVILY_API_KEY"] = tavily_api_key

                tmp_pdf_path = ""
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                    tmp_pdf.write(uploaded_pdf.getvalue())
                    tmp_pdf_path = tmp_pdf.name

                try:
                    with st.spinner("Building FAISS index from PDF..."):
                        index_dir = ingest_pdf(
                            pdf_path=tmp_pdf_path,
                            gemini_api_key=gemini_api_key,
                            index_dir=settings.index_dir,
                        )
                finally:
                    if tmp_pdf_path:
                        Path(tmp_pdf_path).unlink(missing_ok=True)

                if index_dir is None:
                    st.error("Failed to process PDF.")
                else:
                    retriever = load_retriever(
                        gemini_api_key=gemini_api_key,
                        index_dir=index_dir,
                        top_k=settings.default_top_k,
                    )
                    if retriever is None:
                        st.error(
                            "PDF was processed, but retriever loading failed. "
                            "Check Gemini settings and quota."
                        )
                    else:
                        st.session_state.retriever = retriever
                        st.session_state.agent_executor = None
                        st.session_state.agent_error = None
                        st.success("PDF processed. Document search is ready.")

    if gemini_ready:
        os.environ["GEMINI_API_KEY"] = gemini_api_key
    if tavily_ready:
        os.environ["TAVILY_API_KEY"] = tavily_api_key

    _ensure_agent(gemini_api_key=gemini_api_key, model_name=model_name)
    _render_runtime_status(
        gemini_ready=gemini_ready,
        tavily_ready=tavily_ready,
        model_name=model_name,
        retriever_ready=st.session_state.get("retriever") is not None,
        agent_ready=st.session_state.get("agent_executor") is not None,
        agent_error=st.session_state.get("agent_error"),
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input(
        "Ask anything about your docs, latest topics, or general AI..."
    )

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        if st.session_state.agent_executor is None:
            agent_error = st.session_state.get("agent_error")
            if agent_error:
                answer = f"Agent is not available right now: {agent_error}"
            else:
                answer = "Agent is not available right now. Check server-side .env configuration and Gemini quota."
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = run_agent_turn(st.session_state.agent_executor, user_prompt)
                    except Exception as exc:
                        logger.exception("Agent turn failed")
                        if isinstance(exc, ValueError):
                            answer = _sanitize_runtime_error(str(exc))
                        else:
                            answer = "Request failed. Please try again in a moment."
                    st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
