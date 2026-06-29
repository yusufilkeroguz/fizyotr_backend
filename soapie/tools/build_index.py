"""PDF -> chunks.json indeks derleyicisi (OFFLINE, derleme zamani calisir).

Kaynak musculoskeletal examination PDF'ini sayfa sayfa metne cevirir, ~sabit
boyutlu (kelime tabanli) chunk'lara boler ve sayfa metadata'si ile birlikte
soapie/data/chunks.json'a yazar. Boylece Cloud Run runtime'inda pypdf GEREKMEZ
ve cold start hizli kalir (yalnizca hafif JSON yuklenir).

Calistirma:
    .venv/bin/python -m soapie.tools.build_index
    # veya farkli kaynak/cikti:
    .venv/bin/python -m soapie.tools.build_index --pdf "documents/x.pdf"
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = ROOT / "documents" / "muscloskeletal examination.pdf"
DEFAULT_OUT = ROOT / "soapie" / "data" / "chunks.json"

# Chunk boyutu: kelime cinsinden. ~220 kelime ~ 300 token civari; klinik
# muayene metinlerinde bir alt-konuyu (orn. "knee — special tests") tek
# chunk'ta tutmaya yeterli, BM25 icin de yeterince ayirt edici.
WORDS_PER_CHUNK = 220
OVERLAP_WORDS = 40           # chunk sinirinda baglam kaybini azaltir


def _clean(text: str) -> str:
    """Pdf metnindeki kirilmis satir/bosluklari toparlar."""
    text = text.replace("\xad", "")              # soft hyphen
    text = re.sub(r"-\n(\w)", r"\1", text)        # satir sonu tirelemesi: exa-\nmination -> examination
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)   # satir sonlarini bosluga cevir
    text = re.sub(r"\s{2,}", " ", text)           # fazla bosluklari sikistir
    return text.strip()


def _chunk_words(words: list[str], page: int, start_id: int) -> list[dict]:
    out: list[dict] = []
    step = WORDS_PER_CHUNK - OVERLAP_WORDS
    i = 0
    cid = start_id
    while i < len(words):
        piece = words[i:i + WORDS_PER_CHUNK]
        if len(piece) < 25 and out:
            # cok kisa kuyruk parcasini onceki chunk'a yedir
            out[-1]["text"] += " " + " ".join(piece)
            break
        out.append({"id": cid, "page": page, "text": " ".join(piece)})
        cid += 1
        i += step
    return out


def build(pdf_path: Path, out_path: Path) -> dict:
    reader = PdfReader(str(pdf_path))
    chunks: list[dict] = []
    cid = 0
    n_pages = len(reader.pages)
    for pno, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        text = _clean(raw)
        if len(text) < 60:           # bos / gorsel-agirlikli sayfayi atla
            continue
        words = text.split()
        page_chunks = _chunk_words(words, page=pno, start_id=cid)
        chunks.extend(page_chunks)
        cid += len(page_chunks)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": pdf_path.name,
        "pages": n_pages,
        "chunks": len(chunks),
        "words_per_chunk": WORDS_PER_CHUNK,
        "overlap_words": OVERLAP_WORDS,
    }
    payload = {"meta": meta, "chunks": chunks}
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="SOAPIE RAG indeksi derle (PDF -> chunks.json)")
    ap.add_argument("--pdf", default=str(DEFAULT_PDF))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF bulunamadi: {pdf_path}")
    meta = build(pdf_path, Path(args.out))
    print(f"OK -> {args.out}")
    print(f"  kaynak : {meta['source']}")
    print(f"  sayfa  : {meta['pages']}")
    print(f"  chunk  : {meta['chunks']}")


if __name__ == "__main__":
    main()
