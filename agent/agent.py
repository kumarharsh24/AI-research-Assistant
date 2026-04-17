from __future__ import annotations

import logging
import os
import time
from typing import Optional

from langchain_classic.agents import AgentType, initialize_agent
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.memory import build_memory
from agent.tools import make_document_tool, web_search_tool


logger = logging.getLogger(__name__)


SYSTEM_GUIDANCE = (
    "You are an intelligent assistant that can use tools. "
    "If the user asks about uploaded documents, use document_search. "
    "If the user asks for latest or current events, use web_search. "
    "When using document context, answer ONLY from the returned context. "
    "If context is missing, clearly say you do not have enough document evidence. "
    "When you answer from document_search context, include citations like [Chunk 1]."
)

MAX_AGENT_RETRIES = 2
RETRY_BASE_DELAY_SECONDS = 1.0


def _is_model_unavailable(error_text: str) -> bool:
    return "NOT_FOUND" in error_text and "models/" in error_text


def _is_quota_exhausted(error_text: str) -> bool:
    lowered = error_text.lower()
    markers = [
        "quota is exhausted",
        "resource_exhausted",
        "generate_content_free_tier_requests",
        "limit: 0",
    ]
    return any(marker in lowered for marker in markers)


def _is_retryable_error(error_text: str) -> bool:
    lowered = error_text.lower()
    markers = [
        "timeout",
        "timed out",
        "try again",
        "temporarily unavailable",
        "internal",
        "connection reset",
        "503",
        "429",
        "deadline exceeded",
    ]
    return any(marker in lowered for marker in markers)


def build_agent_executor(
    gemini_api_key: str,
    model_name: str = "gemini-2.0-flash",
    retriever: Optional[BaseRetriever] = None,
):
    """Build a conversational ReAct-style agent backed by Gemini."""
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is required to initialize the agent")

    os.environ["GOOGLE_API_KEY"] = gemini_api_key

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=gemini_api_key,
        temperature=0.2,
    )

    tools = [web_search_tool]
    if retriever is not None:
        tools.append(make_document_tool(retriever))

    memory = build_memory()

    try:
        agent_executor = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            agent_kwargs={"system_message": SYSTEM_GUIDANCE},
        )
    except Exception:
        logger.exception("Failed to initialize agent executor")
        raise

    return agent_executor


def route_user_query(user_query: str) -> str:
    """Lightweight routing hint to improve tool selection."""
    q = user_query.lower()

    web_terms = ["latest", "current", "today", "news", "2026", "recent"]
    doc_terms = ["document", "pdf", "file", "according to", "from the paper"]

    if any(term in q for term in web_terms):
        return (
            "Routing hint: this likely needs real-time web data. "
            "Use the web_search tool if needed.\n\n"
            f"User question: {user_query}"
        )

    if any(term in q for term in doc_terms):
        return (
            "Routing hint: this likely needs uploaded document context. "
            "Use the document_search tool if available.\n\n"
            f"User question: {user_query}"
        )

    return user_query


def run_agent_turn(agent_executor, user_query: str) -> str:
    """Run one user turn through the agent and return output text."""
    routed_query = route_user_query(user_query)
    result = None

    for attempt in range(MAX_AGENT_RETRIES + 1):
        try:
            result = agent_executor.invoke({"input": routed_query})
            break
        except Exception as exc:
            err = str(exc)

            if _is_model_unavailable(err):
                raise ValueError(
                    "Configured Gemini model is unavailable for this API key. "
                    "Set GEMINI_MODEL to a currently supported model, e.g. gemini-2.0-flash."
                ) from exc

            if _is_quota_exhausted(err):
                raise ValueError(
                    "Gemini API quota is exhausted for this key/project. "
                    "Check quota/billing, wait for reset, or use another API key."
                ) from exc

            can_retry = attempt < MAX_AGENT_RETRIES and _is_retryable_error(err)
            if can_retry:
                wait_seconds = RETRY_BASE_DELAY_SECONDS * (2**attempt)
                logger.warning(
                    "Agent turn failed (attempt %s/%s). Retrying in %.1fs. Error: %s",
                    attempt + 1,
                    MAX_AGENT_RETRIES + 1,
                    wait_seconds,
                    err,
                )
                time.sleep(wait_seconds)
                continue

            logger.exception("Agent turn execution failed")
            raise ValueError("Agent request failed. Please retry in a moment.") from exc

    if result is None:
        raise ValueError("Agent could not produce a response after retries.")

    if isinstance(result, dict):
        return str(result.get("output", ""))
    return str(result)
