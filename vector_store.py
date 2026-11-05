"""Persistent ChromaDB indexing and cosine-similarity FAQ and notice search."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from document_processing import (
    DEFAULT_FAQ_PATH,
    FAQEntry,
    generate_embeddings,
    load_faq_entries,
)
from notice_manager import load_notices


COLLECTION_NAME = "college_faq"
DEFAULT_PERSIST_DIRECTORY = Path(__file__).parent / "chroma_db"

_CLIENT_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class FAQSearchResult:
    """A retrieved FAQ or notice chunk and its cosine score."""

    question: str
    content: str
    category: str
    source: str
    distance: float
    similarity: float


def get_faq_category(question: str) -> str:
    """Assign a simple category from common FAQ terms."""
    text = question.casefold()
    categories = {
        "Admissions": ("admission", "apply", "application", "enroll"),
        "Fees & payments": ("fee", "tuition", "payment", "account"),
        "Travel tips": ("library", "campus", "opening hours", "hostel", "canteen"),
        "Traveler services": ("traveler id", "id card", "registrar", "attendance", "portal"),
        "Academics": ("academic", "advisor", "course", "degree", "syllabus", "exam"),
        "Events": ("event", "symposium", "workshop", "fest"),
    }
    for category, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "General"


def _get_client_and_collection(persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY):
    """Open or reuse the persistent ChromaDB collection using cosine distance (cached)."""
    import chromadb

    path_str = str(persist_directory)
    if path_str not in _CLIENT_CACHE:
        _CLIENT_CACHE[path_str] = chromadb.PersistentClient(path=path_str)

    client = _CLIENT_CACHE[path_str]
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def _metadata_for_entry(entry: FAQEntry, source: str) -> dict[str, str]:
    category = entry.category if entry.category and entry.category != "General" else get_faq_category(entry.question)
    return {
        "question": entry.question,
        "category": category,
        "source": source,
    }


def rebuild_index(
    faq_path: str | Path = DEFAULT_FAQ_PATH,
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    model: object | None = None,
    include_notices: bool = True,
) -> int:
    """Replace the Karnataka travel knowledge base collection from the source text file and active notices."""
    faq_path = Path(faq_path)
    entries = load_faq_entries(faq_path) if faq_path.exists() else []

    all_ids: list[str] = []
    all_texts: list[str] = []
    all_metadatas: list[dict[str, str]] = []

    # Add FAQ entries
    for index, entry in enumerate(entries):
        all_ids.append(f"faq-{index + 1}")
        all_texts.append(entry.content)
        all_metadatas.append(_metadata_for_entry(entry, faq_path.name))

    # Add active notices
    if include_notices:
        try:
            notices = load_notices()
            for notice in notices:
                if notice.is_active:
                    all_ids.append(f"notice-{notice.id}")
                    all_texts.append(notice.searchable_chunk)
                    all_metadatas.append(
                        {
                            "question": f"Notice: {notice.title}",
                            "category": notice.category,
                            "source": f"Campus Notice ({notice.priority})",
                        }
                    )
        except Exception:
            pass

    if not all_texts:
        return 0

    embeddings = generate_embeddings(all_texts, model=model)

    client, _ = _get_client_and_collection(persist_directory)
    existing_names = {
        collection.name if hasattr(collection, "name") else collection
        for collection in client.list_collections()
    }
    if COLLECTION_NAME in existing_names:
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=all_ids,
        documents=all_texts,
        embeddings=embeddings.tolist(),
        metadatas=all_metadatas,
    )
    return len(all_texts)


def search_faq(
    query: str,
    top_k: int = 3,
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    model: object | None = None,
) -> list[FAQSearchResult]:
    """Retrieve the top FAQ and notice chunks using cosine semantic similarity."""
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    _, collection = _get_client_and_collection(persist_directory)
    count = collection.count()
    if count == 0:
        return []

    query_embedding = generate_embeddings([query], model=model)
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    matches: list[FAQSearchResult] = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    for document, metadata, distance in zip(documents, metadatas, distances):
        distance = float(distance)
        similarity = max(0.0, min(1.0, 1.0 - distance))
        matches.append(
            FAQSearchResult(
                question=metadata["question"],
                content=document,
                category=metadata["category"],
                source=metadata["source"],
                distance=distance,
                similarity=similarity,
            )
        )
    return matches
