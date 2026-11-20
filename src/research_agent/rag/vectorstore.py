"""ChromaDB-backed knowledge base (ingest, search, delete)."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from research_agent.config import Settings
from research_agent.observability.logging import get_logger
from research_agent.rag.loader import SourceDoc, chunk_document, load_directory
from research_agent.schemas import Evidence

log = get_logger("research_agent.kb")


class KnowledgeBase:
    def __init__(self, settings: Settings, embeddings: Embeddings, persist_dir: Path | None = None):
        self.settings = settings
        self.store = Chroma(
            collection_name=settings.collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir or settings.chroma_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._sensitive_terms: set[str] = set(settings.sensitive_terms)

    # ---------------------------------------------------------------- ingest
    def add_document(self, doc: SourceDoc) -> int:
        self.delete_document(doc.doc_id)
        chunks = chunk_document(doc, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            return 0
        ids = [f"{doc.doc_id}::{c.metadata['chunk_index']}" for c in chunks]
        self.store.add_documents(chunks, ids=ids)
        if doc.classification == "confidential":
            self._sensitive_terms.update(doc.sensitive_terms)
        log.info("kb_document_added", doc_id=doc.doc_id, chunks=len(chunks))
        return len(chunks)

    def ingest_directory(self, directory: Path, reset: bool = False) -> dict[str, int]:
        if reset:
            self.reset()
        stats = {}
        for doc in load_directory(directory):
            stats[doc.doc_id] = self.add_document(doc)
        return stats

    def refresh_sensitive_terms(self, directory: Path) -> None:
        for doc in load_directory(directory):
            if doc.classification == "confidential":
                self._sensitive_terms.update(doc.sensitive_terms)

    def delete_document(self, doc_id: str) -> int:
        existing = self.store.get(where={"doc_id": doc_id})
        ids = existing.get("ids", [])
        if ids:
            self.store.delete(ids=ids)
        return len(ids)

    def reset(self) -> None:
        ids = self.store.get().get("ids", [])
        if ids:
            self.store.delete(ids=ids)

    # ---------------------------------------------------------------- query
    def count(self) -> int:
        return len(self.store.get().get("ids", []))

    def list_documents(self) -> list[dict]:
        metas = self.store.get(include=["metadatas"]).get("metadatas", [])
        docs: dict[str, dict] = {}
        for m in metas:
            d = docs.setdefault(m["doc_id"], {**{k: m.get(k) for k in ("doc_id", "title", "classification", "department", "last_updated")}, "chunks": 0})
            d["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["doc_id"])

    def search(self, query: str, k: int | None = None) -> list[Evidence]:
        k = k or self.settings.retrieval_k
        results = self.store.similarity_search_with_relevance_scores(query, k=k)
        out = []
        for doc, score in results:
            m = doc.metadata
            out.append(Evidence(
                kind="internal",
                title=f"{m.get('title')}" + (f" — {m['section'].split(' > ')[-1]}" if m.get("section") else ""),
                content=doc.page_content,
                doc_id=m.get("doc_id"),
                classification=m.get("classification", "internal"),
                last_updated=m.get("last_updated") or None,
                score=round(float(score), 4),
            ))
        return out

    @property
    def sensitive_terms(self) -> set[str]:
        return self._sensitive_terms
