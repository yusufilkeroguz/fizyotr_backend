"""SOAPIE oneri agent'i — pipeline: sorgu cikarimi -> BM25 retrieval -> sentez.

Akis:
  1) extract_query : Turkce subjective -> Ingilizce QueryTerms (cross-lingual koprü)
  2) retrieval     : BM25 ile ilgili PDF chunk'lari getir
  3) synthesize    : yalnizca getirilen baglama dayali SOAP onerisi (taslak)

Model: claude-opus-4-8, adaptive thinking, structured outputs (messages.parse).
ANTHROPIC_API_KEY ortam degiskeninden okunur; gizli anahtar koda gomulmez.
"""
from __future__ import annotations

import os
from typing import Optional

from . import prompts, retrieval
from .schema import QueryTerms, SoapieSuggestion

MODEL = "claude-opus-4-8"

# Cikti her zaman taslaktir — sabit, modele uretttirilmez.
DISCLAIMER = (
    "Bu cikti yapay zeka tarafindan uretilmis bir TASLAKTIR; tani koymaz ve "
    "klinik karar yerine gecmez. Tum oneriler fizyoterapist tarafindan gozden "
    "gecirilip onaylanmalidir. Tedavi/egzersiz plani icin ayrica harici bir "
    "tedavi kaynagi gereklidir."
)


class SoapieError(Exception):
    """Agent calistirilirken olusan kullanima-donuk hata (HTTP 4xx/5xx)."""

    def __init__(self, message: str, status: int = 500):
        super().__init__(message)
        self.status = status


def _client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SoapieError(
            "ANTHROPIC_API_KEY tanimli degil. SOAPIE agent'i Claude API gerektirir; "
            "ortam degiskenini ayarlayin.",
            status=503,
        )
    try:
        import anthropic
    except ImportError:
        raise SoapieError("'anthropic' paketi kurulu degil (pip install anthropic).", status=503)
    return anthropic.Anthropic()


def _query_text(qt: QueryTerms) -> str:
    parts = [qt.region, *qt.english_keywords, *qt.suspected_structures, *qt.symptoms, *qt.red_flags]
    return " ".join(p for p in parts if p)


def extract_query(subjective: str, patient: Optional[dict] = None) -> QueryTerms:
    client = _client()
    user = subjective.strip()
    if patient:
        user += f"\n\n[Hasta bilgisi]: {patient}"
    # Basit cikarim gorevi -> thinking kapali (dusuk gecikme).
    resp = client.messages.parse(
        model=MODEL,
        max_tokens=2000,
        thinking={"type": "disabled"},
        system=prompts.QUERY_SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_format=QueryTerms,
    )
    return resp.parsed_output


def _context_block(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(f"[Kaynak — sayfa {c['page']}]\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def synthesize(subjective: str, chunks: list[dict], patient: Optional[dict] = None) -> SoapieSuggestion:
    client = _client()
    context = _context_block(chunks)
    user = (
        f"# Hastanin subjektif oykusu (terapist girdisi)\n{subjective.strip()}\n"
    )
    if patient:
        user += f"\n# Hasta bilgisi\n{patient}\n"
    user += (
        f"\n# Kaynak muayene kitabindan getirilen ilgili bolumler\n{context}\n"
        "\nYukaridaki kaynak bolumlere ve subjektif oykuye dayanarak SOAP "
        "onerini uret. Kaynakta olmayan bilgiyi uydurma."
    )
    # Klinik akil yurutme gerektirir -> adaptive thinking, ama effort=medium
    # ile gecikme/maliyet dengelenir (kalite korunarak).
    resp = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        system=prompts.SYNTH_SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_format=SoapieSuggestion,
    )
    return resp.parsed_output


def suggest(subjective: str, patient: Optional[dict] = None, k: int = 8) -> dict:
    """Ucu birlestiren ana giris noktasi. Donen dict dogrudan JSON'lanabilir."""
    if not subjective or not subjective.strip():
        raise SoapieError("'subjective' (terapistin girdigi subjektif oyku) zorunludur.", status=400)

    qt = extract_query(subjective, patient)
    chunks = retrieval.search(_query_text(qt), k=k)
    if not chunks:
        raise SoapieError(
            "Kaynakta ilgili bolum bulunamadi; subjektif oykuyu biraz daha "
            "detaylandirin (bolge/semptom).",
            status=422,
        )
    suggestion = synthesize(subjective, chunks, patient)

    return {
        "soapie": suggestion.model_dump(),
        "disclaimer": DISCLAIMER,
        "query_terms": qt.model_dump(),                       # seffaflik / debug
        "retrieved_pages": sorted({c["page"] for c in chunks}),
        "source": retrieval.get_index().meta.get("source"),
    }
