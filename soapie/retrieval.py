"""BM25 lexical retrieval — chunks.json uzerinde.

Indeks bir kez (ilk kullanimda) yuklenir ve sureli onbellege alinir. Saf
Python (rank-bm25); embedding / vektor DB / harici API GEREKMEZ. Sorgu
terimleri Ingilizce'dir (agent.extract_query cross-lingual koprüyü kurar),
boylece Turkce girdi - Ingilizce kaynak dil farki asilir.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

DATA_PATH = Path(__file__).resolve().parent / "data" / "chunks.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class _Index:
    def __init__(self, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.meta: dict = payload["meta"]
        self.chunks: list[dict] = payload["chunks"]
        self._bm25 = BM25Okapi([_tokens(c["text"]) for c in self.chunks])

    def search(self, query: str, k: int = 8) -> list[dict]:
        q = _tokens(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[dict] = []
        for i in order[:k]:
            if scores[i] <= 0:
                break
            c = self.chunks[i]
            out.append({"id": c["id"], "page": c["page"], "score": float(scores[i]), "text": c["text"]})
        return out


_INDEX: Optional[_Index] = None


def get_index() -> _Index:
    """Indeksi (lazy, singleton) dondurur. chunks.json yoksa anlamli hata verir."""
    global _INDEX
    if _INDEX is None:
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"RAG indeksi yok: {DATA_PATH}. Once derle: "
                "python -m soapie.tools.build_index"
            )
        _INDEX = _Index(DATA_PATH)
    return _INDEX


def search(query: str, k: int = 8) -> list[dict]:
    return get_index().search(query, k=k)
