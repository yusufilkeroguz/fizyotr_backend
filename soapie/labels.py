"""Cikti diline gore panel/alan basliklari (UI label'lari).

Backend, frontend'in hardcode etmemesi icin basliklari yaniti dilde dondurur
(secim QueryTerms.language'a gore). Tutarlilik icin sabit sozluk kullanilir;
bilinmeyen dilde Ingilizce'ye duser. Yeni dil eklemek = yeni bir sozluk girisi.
"""
from __future__ import annotations

# Anahtarlar SOAPIE alan adlariyla birebir (frontend dogrudan eslestirir).
_EN = {
    "subjective_followup_questions": "Subjective — Follow-up Questions",
    "objective_exam": "Objective Examination Plan",
    "observation": "Observation",
    "palpation": "Palpation",
    "range_of_motion": "Range of Motion (ROM)",
    "resistive_tests": "Resistive Tests",
    "special_and_neuro_tests": "Special & Neurological Tests",
    "assessment": "Assessment",
    "possible_conditions": "Possible Conditions",
    "clinical_reasoning": "Clinical Reasoning",
    "paradigm_clues": "Diagnostic Clues",
    "plan": "Plan",
    "recommended_tests": "Recommended Tests / Imaging",
    "treatment_note": "Treatment Note",
    "citations": "Source Pages",
    "disclaimer": "Disclaimer",
}

_TR = {
    "subjective_followup_questions": "Sorulması Önerilen Ek Sorular",
    "objective_exam": "Objektif Muayene Planı",
    "observation": "Gözlem",
    "palpation": "Palpasyon",
    "range_of_motion": "Hareket Açıklığı (ROM)",
    "resistive_tests": "Dirençli Testler",
    "special_and_neuro_tests": "Özel ve Nörolojik Testler",
    "assessment": "Değerlendirme",
    "possible_conditions": "Olası Tanılar",
    "clinical_reasoning": "Klinik Akıl Yürütme",
    "paradigm_clues": "Tanı İpuçları",
    "plan": "Plan",
    "recommended_tests": "Önerilen Testler / Görüntüleme",
    "treatment_note": "Tedavi Notu",
    "citations": "Kaynak Sayfalar",
    "disclaimer": "Uyarı",
}

# Dil ADI (QueryTerms.language) -> sozluk. Anahtar kucuk harfle eslestirilir.
_BY_LANG = {"english": _EN, "turkish": _TR, "türkçe": _TR, "ingilizce": _EN}


def panel_labels(language: str | None) -> dict:
    """Verilen dil icin baslik sozlugu; bilinmeyen dilde Ingilizce'ye duser."""
    return dict(_BY_LANG.get((language or "").strip().lower(), _EN))
