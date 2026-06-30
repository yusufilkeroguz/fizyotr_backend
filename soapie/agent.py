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
from .schema import (
    QueryTerms, SoapieSuggestion,
    SubjectiveSection, ObjectiveSection, AssessmentSection, PlanSection,
)

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


def _synth_user(subjective: str, chunks: list[dict], patient: Optional[dict],
                language: Optional[str] = None) -> str:
    context = _context_block(chunks)
    user = ""
    if language:
        # Net dil direktifi — talimat/kaynak dili ne olursa olsun cikti bu dilde.
        user += (f"IMPORTANT — OUTPUT LANGUAGE: Write EVERY field value entirely in "
                 f"{language}. Do not use any other language, regardless of the language "
                 f"of these instructions or the source excerpts.\n\n")
    user += f"# Hastanin subjektif oykusu (terapist girdisi)\n{subjective.strip()}\n"
    if patient:
        user += f"\n# Hasta bilgisi\n{patient}\n"
    user += (
        f"\n# Kaynak muayene kitabindan getirilen ilgili bolumler\n{context}\n"
        "\nYukaridaki kaynak bolumlere ve subjektif oykuye dayanarak SOAP "
        "onerini uret. Kaynakta olmayan bilgiyi uydurma."
    )
    return user


def synthesize(subjective: str, chunks: list[dict], patient: Optional[dict] = None,
               language: Optional[str] = None) -> SoapieSuggestion:
    client = _client()
    # Sonnet 4.6 + adaptive thinking + effort=low: hizli, bu gorevde kalite yeterli.
    # Cikti prompt'ta sinirli (madde sayilari) -> daha az token -> daha hizli.
    resp = client.messages.parse(
        model=SYNTH_MODEL,
        max_tokens=8000,
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
        system=prompts.SYNTH_SYSTEM,
        messages=[{"role": "user", "content": _synth_user(subjective, chunks, patient, language)}],
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
    return _finalize(synthesize(subjective, chunks, patient, qt.language).model_dump(), qt, chunks)


# SOAP'in 4 bolumu AYRI ve PARALEL uretilir; her biri kendi structured ciktisini
# dondurur. (key, pydantic model, "sadece bu bolumu uret" talimati).
_SECTIONS = [
    ("subjective", SubjectiveSection,
     "SADECE 'subjective_followup_questions' (en fazla 5 soru), 'disclaimer' "
     "(oyku diliyle ayni dilde: taslak, tani koymaz, terapist onayi, tedavi icin "
     "harici kaynak) ve 'citations' uret."),
    ("objective", ObjectiveSection,
     "SADECE 'objective_exam' (observation/palpation/range_of_motion/"
     "resistive_tests/special_and_neuro_tests; her biri en fazla 4 madde) ve "
     "'citations' uret."),
    ("assessment", AssessmentSection,
     "SADECE 'assessment' (possible_conditions en fazla 4, clinical_reasoning "
     "2-3 cumle, paradigm_clues en fazla 4) ve 'citations' uret."),
    ("plan", PlanSection,
     "SADECE 'plan' (recommended_tests en fazla 5 + treatment_note: bu kaynak "
     "tedaviyi kapsamaz, harici tedavi kaynagi gerekir) ve 'citations' uret."),
]


def _section_call(client, key, model_cls, instruction, subjective, chunks, patient, language):
    user = _synth_user(subjective, chunks, patient, language) + "\n\n" + instruction
    resp = client.messages.parse(
        model=SYNTH_MODEL,
        max_tokens=3000,
        thinking={"type": "adaptive"},
        output_config={"effort": "low"},
        system=prompts.SYNTH_SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_format=model_cls,
    )
    return key, resp.parsed_output.model_dump()


def stream_synthesis(subjective: str, chunks: list[dict], qt: QueryTerms,
                     patient: Optional[dict] = None):
    """4 bolumu PARALEL uretir; her biri bitince o panele HAZIR structured veri akar.

    Yield edilen dict'ler:
      {"type":"section","key":"subjective|objective|assessment|plan","data":{...}}
      {"type":"result", ...}        tum bolumler birlesik (suggest() ile ayni sekil)
      {"type":"error","error": "..."}
    Bolumler bagimsiz oldugu icin A en sona kalmaz (~10-12s'de gelir), toplam
    sure ~max(bolum) olur (sentezi seri uretmeye gore cok daha hizli).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    client = _client()
    soapie: dict = {}
    disclaimer = ""
    cites: set[int] = set()
    field = {"subjective": "subjective_followup_questions", "objective": "objective_exam",
             "assessment": "assessment", "plan": "plan"}

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(_section_call, client, k, m, ins, subjective, chunks, patient, qt.language)
                for k, m, ins in _SECTIONS]
        for fut in as_completed(futs):
            try:
                key, data = fut.result()
            except Exception as e:
                yield {"type": "error", "error": f"{type(e).__name__}: {e}"}
                continue
            cites.update(data.pop("citations", []) or [])
            if key == "subjective":
                disclaimer = (data.pop("disclaimer", "") or "").strip()
            f = field[key]
            soapie[f] = data[f]
            yield {"type": "section", "key": key, "data": {f: data[f]}}

    soapie["citations"] = sorted(cites) or sorted({c["page"] for c in chunks})
    yield {"type": "result",
           "soapie": soapie,
           "disclaimer": disclaimer or DISCLAIMER,
           "query_terms": qt.model_dump(),
           "retrieved_pages": sorted({c["page"] for c in chunks}),
           "source": retrieval.get_index().meta.get("source")}
