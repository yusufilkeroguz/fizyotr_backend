"""SOAPIE oneri agent'i paketi.

Import edildiginde sibling `soapie/.env` dosyasini (varsa) os.environ'a yukler
— yalnizca LOKAL gelistirme kolayligi icin. Gercek ortam degiskenleri (orn.
Cloud Run / Secret Manager) ASLA ezilmez. Ek bagimlilik yok (python-dotenv
gerekmez). Cloud Run'da .env deploy edilmez; ANTHROPIC_API_KEY'i env olarak set et.

Anahtar adi koprüsü: .env icinde CLAUDE_API_KEY de kabul edilir ve
ANTHROPIC_API_KEY'e eslenir (anthropic SDK bu adi okur).
"""
from __future__ import annotations

import os
from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parent / ".env"


def _load_env(path: Path = _ENV_PATH) -> None:
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:        # gercek env'i ezme
                os.environ[key] = val
    # CLAUDE_API_KEY -> ANTHROPIC_API_KEY koprüsü (SDK ANTHROPIC_API_KEY okur)
    if not os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("CLAUDE_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = os.environ["CLAUDE_API_KEY"]


_load_env()
