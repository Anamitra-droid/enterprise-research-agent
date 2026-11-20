"""Small lexical helpers shared by the offline mock, validators and evaluation."""

from __future__ import annotations

import re

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "for", "with", "at", "by",
    "from", "as", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that",
    "these", "those", "what", "which", "who", "whom", "how", "why", "when", "where", "do", "does",
    "did", "can", "could", "should", "would", "will", "shall", "may", "might", "must", "i", "you",
    "he", "she", "they", "them", "me", "my", "your", "their", "about", "tell", "please", "give",
    "any", "some", "there", "here", "than", "then", "so", "not", "no", "yes", "into", "over",
    "also", "s", "get", "gets", "have", "has", "had", "much", "many", "more", "most", "all",
    "per", "each", "like", "know", "need", "want", "us", "we", "our", "ours", "whats", "what's",
    "vs", "versus", "compare", "compared", "comparison", "against", "between", "currently",
    "current", "latest", "search", "web", "find", "look", "up", "news", "today", "now", "still",
}

_WORD = re.compile(r"[a-z0-9][a-z0-9\-\.%$]*")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
NUMBER = re.compile(r"(?:\$\s?)?\d[\d,]*(?:\.\d+)?\s?(?:%|bps|basis points|million|m\b|bn|billion|k\b)?", re.I)


def tokens(text: str) -> list[str]:
    text = re.sub(r"['’]s\b", "", text.lower())  # possessives: "paynova's" -> "paynova"
    return [t.strip(".-") for t in _WORD.findall(text) if t.strip(".-")]


def content_tokens(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in STOPWORDS and len(t) > 1}


def sentences(text: str) -> list[str]:
    out = []
    for s in _SENT.split(text):
        s = s.strip()
        if s.startswith(("#", "|", "---")):
            continue  # markdown headings / tables / rules are not statements
        s = s.replace("**", "").replace("`", "").strip(" -*\t")
        if len(s) > 25:
            out.append(s)
    return out


def overlap(a: str | set[str], b: str | set[str]) -> float:
    """Fraction of a's content tokens that appear in b."""
    ta = a if isinstance(a, set) else content_tokens(a)
    tb = b if isinstance(b, set) else content_tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def numbers_in(text: str) -> set[str]:
    return {re.sub(r"\s+", "", m.group(0)).lower() for m in NUMBER.finditer(text) if any(c.isdigit() for c in m.group(0))}
