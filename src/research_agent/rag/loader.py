"""Load markdown documents with YAML front-matter and split them into chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import frontmatter
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


@dataclass
class SourceDoc:
    doc_id: str
    title: str
    content: str
    classification: str = "internal"
    department: str = "general"
    last_updated: str | None = None
    sensitive_terms: list[str] = field(default_factory=list)

    @property
    def metadata(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "classification": self.classification,
            "department": self.department,
            "last_updated": self.last_updated or "",
        }


def load_markdown(path: Path) -> SourceDoc:
    post = frontmatter.load(path)
    meta = post.metadata
    return SourceDoc(
        doc_id=str(meta.get("id") or path.stem),
        title=str(meta.get("title") or path.stem.replace("-", " ").title()),
        content=post.content,
        classification=str(meta.get("classification", "internal")),
        department=str(meta.get("department", "general")),
        last_updated=str(meta["last_updated"]) if meta.get("last_updated") else None,
        sensitive_terms=list(meta.get("sensitive_terms", []) or []),
    )


def load_directory(directory: Path) -> list[SourceDoc]:
    return [load_markdown(p) for p in sorted(Path(directory).glob("*.md"))]


def chunk_document(doc: SourceDoc, chunk_size: int = 900, chunk_overlap: int = 150) -> list[Document]:
    """Header-aware split, then size-bounded split. Each chunk carries its section path."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False
    )
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    sections = header_splitter.split_text(doc.content) or [Document(page_content=doc.content)]
    chunks: list[Document] = []
    for section in sections:
        section_name = " > ".join(v for k, v in sorted(section.metadata.items()) if k.startswith("h"))
        for piece in size_splitter.split_text(section.page_content):
            meta = {**doc.metadata, "section": section_name, "chunk_index": len(chunks)}
            chunks.append(Document(page_content=piece, metadata=meta))
    return chunks
