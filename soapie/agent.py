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

# Hiz/kalite/maliyet dengesi:
#  - Sorgu cikarimi basit bir ceviri/cikarim isi -> Haiku (en hizli).
#  - Sentez getirilen metni yapilandirir -> Sonnet 4.6 (Opus'a yakin, ~2-3x hizli).
QUERY_MODEL = "claude-haiku-4-5"
SYNTH_MODEL = "claude-sonnet-4-6"

# Yedek disclaimer (Turkce) — model uretmezse kullanilir. Normalde model,
# oykunun diliyle ayni dilde bir disclaimer uretir (bkz. SYNTH_SYSTEM + schema).
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
    # Basit cikarim gorevi -> Haiku, thinking yok (dusuk gecikme).
    resp = client.messages.parse(
        model=QUERY_MODEL,
        max_tokens=1500,
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


def _synth_user(subjective: str, chunks: list[dict], patient: Optional[dict]) -> str:
    context = _context_block(chunks)
    user = f"# Hastanin subjektif oykusu (terapist girdisi)\n{subjective.strip()}\n"
    if patient:
        user += f"\n# Hasta bilgisi\n{patient}\n"
    user += (
        f"\n# Kaynak muayene kitabindan getirilen ilgili bolumler\n{context}\n"
        "\nYukaridaki kaynak bolumlere ve subjektif oykuye dayanarak SOAP "
        "onerini uret. Kaynakta olmayan bilgiyi uydurma."
    )
    return user


def synthesize(subjective: str, chunks: list[dict], patient: Optional[dict] = None) -> SoapieSuggestion:
    client = _client()
    # Sonnet 4.6 + adaptive thinking + effort=low: hizli, bu gorevde kalite yeterli.
    # Cikti prompt'ta sinirli (madde sayilari) -> daha az token -> daha hizli.
    resp = client.messages.parse(
        model=SYNTH_MODEL,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
        system=prompts.SYNTH_SYSTEM,
        messages=[{"role": "user", "content": _synth_user(subjective, chunks, patient)}],
        output_format=SoapieSuggestion,
    )
    return resp.parsed_output


def prepare(subjective: str, patient: Optional[dict] = None, k: int = 8) -> tuple[QueryTerms, list[dict]]:
    """Sentezden ONCEKI senkron asama: dogrulama + sorgu cikarimi + retrieval.

    HTTP hata kodlari (400/422) icin streaming baslamadan once cagrilir.
    """
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
    return qt, chunks


def _finalize(data: dict, qt: QueryTerms, chunks: list[dict]) -> dict:
    disclaimer = (data.pop("disclaimer", "") or "").strip() or DISCLAIMER
    return {
        "soapie": data,
        "disclaimer": disclaimer,                             # oyku diliyle ayni dil
        "query_terms": qt.model_dump(),                       # seffaflik / debug
        "retrieved_pages": sorted({c["page"] for c in chunks}),
        "source": retrieval.get_index().meta.get("source"),
    }


def suggest(subjective: str, patient: Optional[dict] = None, k: int = 8) -> dict:
    """Tek-parca (non-streaming) giris noktasi. Donen dict dogrudan JSON'lanabilir."""
    qt, chunks = prepare(subjective, patient, k)
    return _finalize(synthesize(subjective, chunks, patient).model_dump(), qt, chunks)


def stream_synthesis(subjective: str, chunks: list[dict], qt: QueryTerms,
                     patient: Optional[dict] = None):
    """Sentezi parca parca akitir (SSE icin generator).

    Yield edilen dict'ler:
      {"type":"delta","text": "..."}     model metni aktikça
      {"type":"result", ...}             tam, dogrulanmis sonuc (suggest() ile ayni sekil)
      {"type":"error","error": "..."}    uretim/parse hatasi
    Not: streaming yolunda structured-output yerine prompt-ile-JSON kullanilir;
    sonunda json.loads + pydantic (SoapieSuggestion) ile dogrulanir.
    """
    import json
    client = _client()
    user = _synth_user(subjective, chunks, patient) + "\n\n" + prompts.JSON_INSTRUCTION
    parts: list[str] = []
    try:
        with client.messages.stream(
            model=SYNTH_MODEL,
            max_tokens=8000,
            thinking={"type": "adaptive"},
            output_config={"effort": "low"},
            system=prompts.SYNTH_SYSTEM,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            for text in stream.text_stream:
                parts.append(text)
                yield {"type": "delta", "text": text}
        raw = "".join(parts).strip()
        if raw.startswith("```"):                      # olasi kod-bloğu sarmalini ayikla
            raw = raw.strip("`")
            raw = raw[raw.find("{"): raw.rfind("}") + 1]
        sug = SoapieSuggestion(**json.loads(raw))       # dogrula
        yield {"type": "result", **_finalize(sug.model_dump(), qt, chunks)}
    except Exception as e:
        yield {"type": "error", "error": f"{type(e).__name__}: {e}"}
