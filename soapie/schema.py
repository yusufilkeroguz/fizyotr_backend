"""SOAPIE agent'i icin yapilandirilmis cikti (structured output) semalari.

Claude `messages.parse(output_format=...)` ile bu Pydantic modellerine birebir
uyan JSON dondurur. JSON-schema kisitlari nedeniyle alanlar yalin tutuldu
(min/max uzunluk, sayisal kisit YOK; yalnizca str / list / nested object).
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


# ── Adim 1: Turkce subjective -> Ingilizce klinik sorgu terimleri ──────────
class QueryTerms(BaseModel):
    """Cross-lingual koprü: terapistin Turkce metninden BM25 icin EN terimler."""

    region: str = Field(description="Etkilenen anatomik bolge, Ingilizce (orn. 'knee', 'cervical spine', 'shoulder').")
    english_keywords: List[str] = Field(description="PDF (Ingilizce muayene kitabi) icinde aranacak anahtar terimler.")
    suspected_structures: List[str] = Field(description="Suphelenilen yapilar, Ingilizce (orn. 'ACL', 'medial meniscus', 'supraspinatus').")
    symptoms: List[str] = Field(description="Semptomlar, Ingilizce (orn. 'pain', 'instability', 'locking').")
    red_flags: List[str] = Field(description="Varsa kirmizi bayrak/uyari isaretleri, Ingilizce. Yoksa bos liste.")


# ── Adim 3: getirilen baglama dayali SOAP onerisi ──────────────────────────
class ObjectiveExam(BaseModel):
    observation: List[str] = Field(description="Gozlem onerileri (durus, yuruyus, asimetri, deformite).")
    palpation: List[str] = Field(description="Palpasyon onerileri (hassasiyet, sislik, isi).")
    range_of_motion: List[str] = Field(description="Aktif/pasif hareket acikligi ve gonyometrik olcum onerileri.")
    resistive_tests: List[str] = Field(description="Direncli kas kuvveti testi onerileri.")
    special_and_neuro_tests: List[str] = Field(description="Ozel bag/meniskus testleri ve norolojik (refleks/duyu/motor) test onerileri.")


class Assessment(BaseModel):
    possible_conditions: List[str] = Field(description="Olasi tani/klinik durumlar (kaynaktaki paradigmalara dayali).")
    clinical_reasoning: str = Field(description="S ve O verisini birlestiren kisa klinik akil yurutme.")
    paradigm_clues: List[str] = Field(description="Tani(lar)a goturen fiziksel muayene ipuclari.")


class Plan(BaseModel):
    recommended_tests: List[str] = Field(description="Tani netlestirmek icin yapilmasi onerilen ek testler/goruntuleme.")
    treatment_note: str = Field(description="Tedavi/egzersiz plani. Kaynak bunu kapsamadigi icin harici tedavi kaynagi gerektigini belirt.")


class SoapieSuggestion(BaseModel):
    """Agent'in ana ciktisi — terapist onerisi (taslak)."""

    subjective_followup_questions: List[str] = Field(description="Terapistin S'yi tamamlamak icin sormasi onerilen ek sorular (Box 2.1 tarzi).")
    objective_exam: ObjectiveExam
    assessment: Assessment
    plan: Plan
    citations: List[int] = Field(description="Onerilerin dayandirildigi kaynak PDF sayfa numaralari.")
