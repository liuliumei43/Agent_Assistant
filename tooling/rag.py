from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".py",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
}

SKIP_DIRS = {".git", ".venv", "__pycache__", ".runs", "node_modules", "dist", "build"}
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}|[\u4e00-\u9fff]")


@dataclass(frozen=True)
class RagDocument:
    path: str
    text: str
    tokens: Counter[str]


@dataclass(frozen=True)
class RagHit:
    path: str
    score: float
    snippet: str


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _safe_read(path: Path, max_bytes: int) -> str:
    raw = path.read_bytes()[:max_bytes]
    return raw.decode("utf-8", errors="replace")


class LocalRagIndex:
    def __init__(self, workspace: Path, *, max_files: int = 200, max_bytes: int = 64_000) -> None:
        self.workspace = workspace.resolve()
        self.max_files = max_files
        self.max_bytes = max_bytes
        self.documents: list[RagDocument] = []
        self.idf: dict[str, float] = {}

    def build(self) -> None:
        docs: list[RagDocument] = []
        for path in self.workspace.rglob("*"):
            if len(docs) >= self.max_files:
                break
            if any(part in SKIP_DIRS for part in path.relative_to(self.workspace).parts):
                continue
            if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            try:
                text = _safe_read(path, self.max_bytes)
            except OSError:
                continue
            tokens = Counter(tokenize(text))
            if not tokens:
                continue
            docs.append(RagDocument(str(path.relative_to(self.workspace)), text, tokens))
        self.documents = docs
        self.idf = self._build_idf(docs)

    def search(self, query: str, top_k: int = 5) -> list[RagHit]:
        if not self.documents:
            self.build()
        query_tokens = Counter(tokenize(query))
        if not query_tokens:
            return []
        scored: list[RagHit] = []
        for doc in self.documents:
            score = self._cosine(query_tokens, doc.tokens)
            if score <= 0:
                continue
            scored.append(RagHit(doc.path, score, self._snippet(doc.text, query_tokens)))
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    def _build_idf(self, docs: list[RagDocument]) -> dict[str, float]:
        doc_count = max(len(docs), 1)
        df: Counter[str] = Counter()
        for doc in docs:
            df.update(doc.tokens.keys())
        return {term: math.log((doc_count + 1) / (freq + 1)) + 1 for term, freq in df.items()}

    def _cosine(self, query: Counter[str], doc: Counter[str]) -> float:
        terms = set(query) | set(doc)
        dot = 0.0
        query_norm = 0.0
        doc_norm = 0.0
        for term in terms:
            weight = self.idf.get(term, 1.0)
            q = query.get(term, 0) * weight
            d = doc.get(term, 0) * weight
            dot += q * d
            query_norm += q * q
            doc_norm += d * d
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        return dot / math.sqrt(query_norm * doc_norm)

    def _snippet(self, text: str, query: Counter[str]) -> str:
        lowered = text.lower()
        positions = [lowered.find(term) for term in query if lowered.find(term) >= 0]
        start = max(min(positions) - 120, 0) if positions else 0
        snippet = text[start : start + 500]
        return snippet.replace("\n", " ").strip()
