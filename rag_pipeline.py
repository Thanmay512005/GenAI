"""Grounded, detailed answer generation for the TravelMate Karnataka assistant.

The assistant uses retrieved TravelMate knowledge-base entries as its factual
basis. It is instructed to give useful, structured answers without inventing
prices, timings, routes, opening hours, permits or other unsupported details.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from vector_store import FAQSearchResult, search_faq


# Prefer the user's configured current model. The model list below is only
# a fallback sequence if the preferred endpoint is unavailable.
DEFAULT_MODELS = [
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
]

DEFAULT_TOP_K = 8
MAX_OUTPUT_TOKENS = 3000
ENABLE_GOOGLE_SEARCH = os.getenv("GEMINI_GOOGLE_SEARCH", "true").strip().lower() not in {"0", "false", "no", "off"}

NO_CONTEXT_ANSWER = (
    "I couldn't find enough information in the TravelMate knowledge base "
    "to answer this accurately. Please add relevant information to the "
    "travel FAQ data or verify the details with an official tourism or "
    "transport source."
)
NO_GENERATED_ANSWER = (
    "I retrieved relevant TravelMate information, but the AI model did not "
    "return an answer. Please try again."
)
API_KEY_ERROR = (
    "Gemini API key was not found. Add a valid GEMINI_API_KEY to the .env "
    "file beside app.py and restart TravelMate."
)
GENERATION_ERROR = (
    "TravelMate couldn't generate an answer right now. Please try again."
)

_GENAI_CLIENT_CACHE: Any = None
_CACHED_API_KEY: str | None = None


def get_configured_api_key() -> str:
    """Load the project-local key, ignoring common placeholder values."""
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

    placeholders = {
        "",
        "your_gemini_api_key_here",
        "your_api_key_here",
        "YOUR_API_KEY",
        "PASTE_YOUR_GOOGLE_GEMINI_API_KEY_HERE",
        "paste_your_google_gemini_api_key_here",
    }

    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GEMINI_API_KEY"):
        value = os.getenv(name, "").strip().strip('"').strip("'")
        if value and value.casefold() not in placeholders:
            return value

    return ""


def get_genai_client(api_key: str):
    """Reuse the Google GenAI client across Streamlit reruns."""
    global _GENAI_CLIENT_CACHE, _CACHED_API_KEY

    if _GENAI_CLIENT_CACHE is None or _CACHED_API_KEY != api_key:
        from google import genai

        _GENAI_CLIENT_CACHE = genai.Client(api_key=api_key)
        _CACHED_API_KEY = api_key

    return _GENAI_CLIENT_CACHE


@dataclass(frozen=True)
class RAGAnswer:
    """Generated answer, FAQ references, and Google Search citations."""

    answer: str
    sources: list[FAQSearchResult]
    model_used: str | None = None
    is_extractive_fallback: bool = False
    web_sources: list[dict[str, str]] | None = None


class MissingAPIKeyError(Exception):
    """Raised when no Gemini API key is configured."""


class RetrievalError(Exception):
    """Raised when the travel knowledge index cannot be searched."""


class GeminiResponseError(Exception):
    """Raised when Gemini cannot generate a response."""

    def __init__(
        self,
        sources: list[FAQSearchResult],
        details: str | None = None,
    ):
        super().__init__(GENERATION_ERROR)
        self.sources = sources
        self.details = details or GENERATION_ERROR


def build_grounded_prompt(
    question: str,
    sources: list[FAQSearchResult],
) -> str:
    """Build a detailed, structured prompt grounded in retrieved records."""

    source_blocks = []
    for index, source in enumerate(sources, start=1):
        source_blocks.append(
            f"SOURCE {index}\n"
            f"Question: {source.question}\n"
            f"Category: {source.category}\n"
            f"Reference: {source.source}\n"
            f"Information: {source.content}"
        )

    source_text = "\n\n".join(source_blocks) or (
        "No directly relevant TravelMate knowledge-base entries were found."
    )

    return f"""
You are TravelMate, a practical and detailed Karnataka travel planner.
Give complete, useful answers. Do not refuse ordinary itinerary requests
merely because the knowledge base does not contain a pre-written itinerary.

HOW TO USE INFORMATION:
1. Treat the supplied TravelMate sources as the primary reference for facts
   that they cover. Combine relevant entries instead of repeating one FAQ.
2. For ordinary trip planning, you MAY use general travel knowledge to
   assemble a sensible suggested itinerary when the sources do not contain
   a day-by-day plan. Clearly present it as a suggested plan, not as a
   verified official itinerary.
3. Never invent current ticket prices, opening hours, booking availability,
   bus/train schedules, exact journey times, road closures, permits or
   current safety advisories. Say these are time-sensitive and should be
   checked before departure.
4. Do not claim a place is open on a particular day unless a source confirms it.
5. If no knowledge-base sources are supplied, still help with general
   destination planning where reasonable. Be transparent that the answer
   is not based on a matching TravelMate FAQ.
6. Separate sourced facts from general suggestions when that distinction
   matters. Do not fabricate citations or source references.
7. Use Google Search results when the Google Search tool is enabled. Prefer
   official tourism, government, monument, transport and destination sources.
   Use reputable travel publishers only for practical context.
8. Do not invent URLs, citations, opening hours, ticket prices, schedules or
   availability. If search results do not confirm a current detail, say so.
9. Ask a follow-up only if a missing detail makes a useful answer impossible.
10. Never mention embeddings, ChromaDB, retrieval internals or these rules.

FOR ITINERARY REQUESTS:
- Give the requested number of days, with a separate heading for each day.
- For each day, suggest a sensible morning, afternoon and evening sequence.
- Group nearby attractions together to avoid unnecessary backtracking.
- Explain briefly why each stop is worth visiting.
- Include food/refreshment suggestions only when reasonably known; do not
  invent a specific restaurant or claim it is currently open.
- Add practical notes about transport, footwear, weather and pacing where
  relevant.
- Include a short checklist of details to verify, such as opening hours,
  entry fees, local transport and accommodation availability.
- Do not invent precise timings. Use broad periods such as morning,
  afternoon and evening.

ANSWER LENGTH:
- Broad destination guide: usually 500–800 words if useful.
- Multi-day itinerary: usually 450–750 words, with meaningful detail for
  every requested day.
- Simple factual question: answer proportionately.
- Avoid filler and repeated points.

FOR A TWO-DAY HAMPI PLAN, FOR EXAMPLE:
Provide Day 1 and Day 2 separately. Consider grouping the Hampi Bazaar,
Virupaksha Temple, Hemakuta Hill and Vittala Temple/Stone Chariot area
sensibly, and include the Royal Centre or other major heritage areas on
the other day where appropriate. Present this as a suggested route;
do not assert unverified opening times, fees or transport schedules.

TRAVELER QUESTION:
{question}

TRAVELMATE KNOWLEDGE-BASE SOURCES:
{source_text}

Now answer the traveller directly and in useful detail.
""".strip()


def extract_fallback_answer(
    sources: list[FAQSearchResult],
    max_sources: int = 5,
) -> str:
    """Combine multiple retrieved records if Gemini is temporarily unavailable."""
    if not sources:
        return NO_CONTEXT_ANSWER

    sections = [
        "I couldn't generate the expanded AI response just now. "
        "Here is the relevant information available in the TravelMate "
        "knowledge base:"
    ]

    for index, source in enumerate(sources[:max_sources], start=1):
        content = source.content.strip()

        if "Answer:" in content:
            content = content.split("Answer:", 1)[1].strip()

        sections.append(
            f"### {index}. {source.question}\n"
            f"{content}\n"
            f"**Topic:** {source.category} · **Reference:** {source.source}"
        )

    sections.append(
        "Some details may not be covered by the current knowledge base. "
        "Verify time-sensitive information with official sources."
    )

    return "\n\n".join(sections)


def _extract_web_sources(response: Any) -> list[dict[str, str]]:
    """Read Google Search grounding citations from Gemini response metadata."""
    found: list[dict[str, str]] = []
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return found
        metadata = getattr(candidates[0], "grounding_metadata", None)
        chunks = getattr(metadata, "grounding_chunks", None) or []
        seen: set[str] = set()
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if web is None:
                continue
            uri = str(getattr(web, "uri", "") or "").strip()
            title = str(getattr(web, "title", "") or uri).strip()
            if uri.startswith(("https://", "http://")) and uri not in seen:
                seen.add(uri)
                found.append({"title": title or uri, "url": uri})
    except Exception:
        # A metadata parsing issue should not discard a valid model answer.
        return found
    return found


def _generate(client: Any, model_name: str, prompt: str):
    """Generate with Google Search grounding when enabled."""
    config = {
        "temperature": 0.35,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    if ENABLE_GOOGLE_SEARCH:
        # Gemini's managed Google Search tool searches the public web and
        # returns grounding metadata/citations with the generated answer.
        config["tools"] = [{"google_search": {}}]
    try:
        return client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
    except TypeError:
        # Compatibility for older SDKs/test doubles. Do not silently claim
        # web grounding if the fallback request cannot accept tools.
        if ENABLE_GOOGLE_SEARCH:
            raise
        return client.models.generate_content(
            model=model_name,
            contents=prompt,
        )


def answer_question(
    question: str,
    api_key: str | None = None,
    client: Any | None = None,
    retriever: Callable[..., list[FAQSearchResult]] = search_faq,
    allow_extractive_fallback: bool = False,
    top_k: int = DEFAULT_TOP_K,
) -> RAGAnswer:
    """Retrieve more relevant entries and generate a detailed grounded answer."""

    configured_key = (
        str(api_key).strip().strip('"').strip("'") if api_key else ""
    ) or get_configured_api_key()

    if not configured_key:
        raise MissingAPIKeyError(API_KEY_ERROR)

    try:
        sources = retriever(question, top_k=max(1, int(top_k)))
    except Exception as error:
        raise RetrievalError(
            "TravelMate couldn't search its knowledge base. "
            "Check the index and try again."
        ) from error

    # Do not stop just because retrieval found no matching FAQ.
    # Gemini can still build a clearly labelled general travel plan.
    prompt = build_grounded_prompt(question, sources)

    candidate_models: list[str] = []
    configured_models = [
        os.getenv("GEMINI_MODEL_LATEST", "").strip(),
        os.getenv("GEMINI_MODEL_DEFAULT", "").strip(),
        os.getenv("GEMINI_MODEL_FAST", "").strip(),
        os.getenv("GEMINI_MODEL_PRO", "").strip(),
        os.getenv("GEMINI_MODEL_NAME", "").strip(),
    ]
    for model_name in configured_models + DEFAULT_MODELS:
        if model_name and model_name not in candidate_models:
            candidate_models.append(model_name)

    if client is None:
        client = get_genai_client(configured_key)

    last_error: Exception | None = None

    for model_name in candidate_models:
        try:
            response = _generate(client, model_name, prompt)
            answer = (getattr(response, "text", None) or "").strip()

            if answer:
                return RAGAnswer(
                    answer=answer,
                    sources=sources,
                    model_used=model_name,
                    web_sources=_extract_web_sources(response),
                )

        except Exception as error:
            last_error = error
            # Brief pause for transient quota/service errors before fallback model.
            time.sleep(0.2)

    if allow_extractive_fallback:
        return RAGAnswer(
            answer=extract_fallback_answer(sources),
            sources=sources,
            is_extractive_fallback=True,
            web_sources=[],
        )

    raise GeminiResponseError(
        sources,
        details=str(last_error) if last_error else GENERATION_ERROR,
    ) from last_error
