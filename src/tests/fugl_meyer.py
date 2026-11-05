"""Fugl-Meyer Üst Ekstremite (FMA-UE) — MediaPipe analiz modülü.

Backend konvansiyonuna (src/tests/*) uyarlanmış Fugl-Meyer otomasyonu.
Diğer test analizcileri gibi tek bir `analyze(v, ...)` fonksiyonu sunar; `v`
bir video yolu veya tarayıcıda çıkarılmış pose+el CSV'sidir.

Pose verisi `pose_common.iter_pose_hands()` üzerinden okunur — 33 BlazePose
gövde landmark'ı + (varsa) 21'er noktalı sol/sağ el. Tüm hesaplar PİKSEL
uzayındadır (normalize koordinatlar W×H ile ölçeklenir).

Kapsanan testler
----------------
Bölüm II  (Sinerjik hareket)  : flexor, extensor
Bölüm III                     : FM-6 (omuz fleks 0-90), FM-7 (pron/sup, dirsek 90)
Bölüm IV                      : FM-8 (omuz abd 0-90), FM-9 (omuz fleks 90-180),
                                FM-10 (pron/sup, dirsek 0)
Bölüm B  (El bileği)          : FM-11, FM-12 (stabilite + tekrarlı F/E),
                                FM-13 (sirkümdüksiyon)
Bölüm D  (Koordinasyon/Hız)   : FM-D (parmak-burun 5×; opsiyonel paired)

`fugl-mayer/fma_mediapipe_analysis.py` tek-dosya prototipinden portlanmıştır.
Port sırasında düzeltilen yapısal hatalar:
  * Orta hat geçişi artık taraf-duyarlı (sağ-etkilenen kolda yanlış pozitif
    veren sabit `<` koşulu kaldırıldı; midline_approach tabanlı).
  * FM-8 supinasyon kompansasyon eşiği ulaşılabilir aralığa çekildi
    (forearm_rot ∈ [-90,90] olduğu için eski 90° eşiği asla tetiklenmiyordu).
  * Sinerji puanı artık aynı kareden (eşzamanlı bitiş pozu) hesaplanıyor;
    farklı karelerin bağımsız peak'lerini birleştirmiyor (skor şişmesi yok).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from src.pose_common import (
    iter_pose_hands,
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    HAND_WRIST, HAND_INDEX_MCP, HAND_MIDDLE_MCP, HAND_INDEX_TIP,
    angle_deg as joint_angle,
)


# =============================================================================
# 1) PİKSEL-UZAYI KARE TİPİ
# =============================================================================

@dataclass
class PoseFrame:
    """Tek kare: piksel uzayında gövde (33) + opsiyonel sol/sağ el (21)."""
    body:   np.ndarray            # (33,2) piksel, eksik = NaN
    bvis:   np.ndarray            # (33,)  visibility
    lhand:  Optional[np.ndarray]  # (21,2) piksel veya None
    rhand:  Optional[np.ndarray]  # (21,2) piksel veya None
    lhvis:  Optional[np.ndarray]
    rhvis:  Optional[np.ndarray]

    def body_xy(self, idx: int, thr: float = 0.3) -> Optional[np.ndarray]:
        if not (0 <= idx < 33):
            return None
        if self.bvis[idx] < thr:
            return None
        p = self.body[idx]
        if not np.isfinite(p).all():
            return None
        return p

    def hand_xy(self, side: str, idx: int, thr: float = 0.3) -> Optional[np.ndarray]:
        hand = self.lhand if side == "left" else self.rhand
        vis  = self.lhvis if side == "left" else self.rhvis
        if hand is None or vis is None or not (0 <= idx < len(hand)):
            return None
        if vis[idx] < thr:
            return None
        p = hand[idx]
        if not np.isfinite(p).all():
            return None
        return p


def _to_px(lm, W: int, H: int):
    """_LM listesini (normalize) piksel (N,2) + visibility (N,) dizisine çevir."""
    n = len(lm)
    xy = np.full((n, 2), np.nan)
    vis = np.zeros(n)
    for i, p in enumerate(lm):
        xy[i] = (p.x * W, p.y * H)
        vis[i] = getattr(p, "visibility", 1.0)
    return xy, vis


def load_frames(v: str, model: str = "heavy") -> tuple[list[PoseFrame], float]:
    """Video/CSV'yi PoseFrame listesine çevir. (frames, fps) döner.
    Pose bulunmayan kareler de listede yer alır (zaman ekseni korunur)."""
    frames: list[PoseFrame] = []
    fps = 30.0
    for fi, t, l2, l3, lh, rh, W, H, f in iter_pose_hands(v, model=model):
        fps = f or fps
        if l2 is None:
            body = np.full((33, 2), np.nan)
            bvis = np.zeros(33)
        else:
            body, bvis = _to_px(l2, W, H)
        lhand = lhvis = rhand = rhvis = None
        if lh is not None:
            lhand, lhvis = _to_px(lh, W, H)
        if rh is not None:
            rhand, rhvis = _to_px(rh, W, H)
        frames.append(PoseFrame(body, bvis, lhand, rhand, lhvis, rhvis))
    return frames, fps


# =============================================================================
# 2) GEOMETRİ / KLİNİK AÇI YARDIMCILARI
# =============================================================================

def _vec_angle_horizontal(v: np.ndarray) -> float:
    """Vektörün +x ekseniyle yaptığı açı (0..180°)."""
    n = float(np.linalg.norm(v)) + 1e-8
    cos_a = float(np.clip(v[0] / n, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def shoulder_abduction_angle(shoulder, elbow, hip) -> float:
    """Glenohumeral abduksiyon: omuz→dirsek ile gövde (omuz→kalça) açısı.
    0°=kol asılı, 90°=yatay, 180°=baş üstü."""
    trunk = np.asarray(hip) - np.asarray(shoulder)
    arm = np.asarray(elbow) - np.asarray(shoulder)
    n = (np.linalg.norm(trunk) * np.linalg.norm(arm)) + 1e-8
    cos_a = float(np.clip(np.dot(trunk, arm) / n, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def elbow_flexion_amount(shoulder, elbow, wrist) -> float:
    """Dirsek fleksiyon miktarı: 180° − eklem açısı. 0=düz, ~150=tam bükük."""
    return 180.0 - joint_angle(shoulder, elbow, wrist)


def forearm_rotation_angle(hand_wrist, index_mcp, middle_mcp) -> float:
    """Önkol rotasyonu (2B). +90≈tam supinasyon, −90≈tam pronasyon, 0≈nötr."""
    w = np.asarray(hand_wrist); i = np.asarray(index_mcp); m = np.asarray(middle_mcp)
    palm = (i + m) / 2.0 - w
    angle = _vec_angle_horizontal(palm)
    sign = 1.0 if i[1] < m[1] else -1.0
    return sign * (90.0 - angle)


def shoulder_elevation_diff_px(left_sh, right_sh, affected_side: str) -> float:
    """Omuz elevasyon farkı (px). Y aşağı pozitif → pozitif = etkilenen daha yukarıda."""
    l = np.asarray(left_sh); r = np.asarray(right_sh)
    return float(r[1] - l[1]) if affected_side == "left" else float(l[1] - r[1])


def midline_approach_ratio(wrist, affected_shoulder, left_hip, right_hip) -> float:
    """Etkilenen bilek orta hatta ne oranda yaklaştı (taraf-duyarlı, normalize).
    0=hareket yok, 1=orta hatta vardı, >1=geçti."""
    midline_x = (float(left_hip[0]) + float(right_hip[0])) / 2.0
    shoulder_x = float(affected_shoulder[0])
    full = midline_x - shoulder_x
    if abs(full) < 1e-6:
        return 0.0
    return (float(wrist[0]) - shoulder_x) / full


def symmetry_score(affected_angle: float, healthy_angle: float,
                   margin_deg: float = 15.0) -> int:
    """İki taraf karşılaştırması → 0/1/2."""
    if affected_angle < 5.0:
        return 0
    if affected_angle >= (healthy_angle - margin_deg):
        return 2
    return 1


def _side_idx(side: str) -> dict:
    if side == "left":
        return {"shoulder": LEFT_SHOULDER, "elbow": LEFT_ELBOW, "wrist": LEFT_WRIST,
                "hip": LEFT_HIP, "hand": "left"}
    return {"shoulder": RIGHT_SHOULDER, "elbow": RIGHT_ELBOW, "wrist": RIGHT_WRIST,
            "hip": RIGHT_HIP, "hand": "right"}


def detect_active_side(frames: Sequence[PoseFrame], thr: float = 0.3) -> str:
    """Daha çok hareket eden bilek = test edilen (etkilenen) taraf."""
    lw, rw = [], []
    for f in frames:
        l = f.body_xy(LEFT_WRIST, thr)
        r = f.body_xy(RIGHT_WRIST, thr)
        if l is not None:
            lw.append(l)
        if r is not None:
            rw.append(r)
    var_l = float(np.var(lw, axis=0).sum()) if len(lw) > 1 else 0.0
    var_r = float(np.var(rw, axis=0).sum()) if len(rw) > 1 else 0.0
    return "left" if var_l > var_r else "right"


# =============================================================================
# 3) ZAMAN SERİSİ EKLEM AÇILARI
# =============================================================================

def series_arm_angles(frames: list[PoseFrame], side: str, thr: float = 0.3) -> dict:
    """Bir kol için frame-frame açı/koordinat serileri (NaN = veri yok)."""
    s = _side_idx(side)
    T = len(frames)
    sh_abd = np.full(T, np.nan)
    el_jt = np.full(T, np.nan)
    fa_rot = np.full(T, np.nan)
    wr_fl = np.full(T, np.nan)
    ua_len = np.full(T, np.nan)
    wrist_xy = np.full((T, 2), np.nan)
    elbow_xy = np.full((T, 2), np.nan)
    shoulder_xy = np.full((T, 2), np.nan)
    valid = np.zeros(T, dtype=bool)

    for t, f in enumerate(frames):
        sh = f.body_xy(s["shoulder"], thr)
        el = f.body_xy(s["elbow"], thr)
        wr = f.body_xy(s["wrist"], thr)
        hp = f.body_xy(s["hip"], thr)
        if sh is not None:
            shoulder_xy[t] = sh
        if el is not None:
            elbow_xy[t] = el
        if wr is not None:
            wrist_xy[t] = wr
        if sh is not None and el is not None:
            ua_len[t] = float(np.linalg.norm(el - sh))
        if sh is not None and el is not None and hp is not None:
            sh_abd[t] = shoulder_abduction_angle(sh, el, hp)
        if sh is not None and el is not None and wr is not None:
            el_jt[t] = joint_angle(sh, el, wr)

        hw = f.hand_xy(s["hand"], HAND_WRIST, thr)
        imc = f.hand_xy(s["hand"], HAND_INDEX_MCP, thr)
        mmc = f.hand_xy(s["hand"], HAND_MIDDLE_MCP, thr)
        if hw is not None and imc is not None and mmc is not None:
            fa_rot[t] = forearm_rotation_angle(hw, imc, mmc)
        if el is not None and wr is not None and mmc is not None:
            if np.linalg.norm(wr - el) > 1.0 and np.linalg.norm(mmc - wr) > 1.0:
                wr_fl[t] = 180.0 - joint_angle(el, wr, mmc)

        valid[t] = (sh is not None and el is not None and wr is not None)

    return dict(shoulder_abd=sh_abd, elbow_joint=el_jt, forearm_rot=fa_rot,
                wrist_flex=wr_fl, upper_arm_len=ua_len, wrist_xy=wrist_xy,
                elbow_xy=elbow_xy, shoulder_xy=shoulder_xy, valid=valid)


def _peak_motion_window(values: np.ndarray, frac: float = 0.2) -> tuple[int, int]:
    arr = values.copy()
    arr[np.isnan(arr)] = -np.inf
    if not np.isfinite(arr).any():
        return 0, len(arr)
    peak = int(np.argmax(arr))
    n = len(arr)
    half = max(1, int(n * frac))
    return max(0, peak - half), min(n, peak + half)


def _shoulder_flexion_proxy_2d(shoulder, elbow, ref_len: float) -> float:
    """Sagital omuz fleksiyonu için 2B foreshortening yaklaşığı (0..180°)."""
    sh = np.asarray(shoulder); el = np.asarray(elbow)
    cur = float(np.linalg.norm(el - sh))
    ratio = min(1.0, cur / max(ref_len, 1e-6))
    flex = float(np.degrees(np.arccos(ratio)))
    dy = el[1] - sh[1]
    if dy < 0:  # dirsek omuzun üzerinde → 90°+ aralık
        vert = min(1.0, abs(dy) / max(ref_len, 1e-6))
        flex = 90.0 + 90.0 * vert
    return float(np.clip(flex, 0.0, 180.0))


# =============================================================================
# 4) SONUÇ TİPİ + BASE ANALYZER
# =============================================================================

@dataclass
class FMTestResult:
    test_id: str
    test_name: str
    section: str
    score: int
    max_score: int
    side: str
    sub_scores: dict = field(default_factory=dict)
    measurements: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    auto: bool = True

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id, "test_name": self.test_name,
            "section": self.section, "score": self.score,
            "max_score": self.max_score, "side": self.side,
            "sub_scores": self.sub_scores, "measurements": self.measurements,
            "notes": self.notes, "auto": self.auto,
        }


class BaseAnalyzer:
    test_id = ""
    test_name = ""
    section = ""
    max_score = 2

    def analyze(self, frames: list[PoseFrame], side: str, fps: float) -> FMTestResult:
        raise NotImplementedError

    def _result(self, score, side, **kw) -> FMTestResult:
        return FMTestResult(self.test_id, self.test_name, self.section,
                            score, self.max_score, side, **kw)


def _have(*xs) -> bool:
    """Tüm argümanlar None değil mi? (numpy dizilerinde `None in (...)`
    'ambiguous truth value' hatası verdiği için açık kontrol gerekir.)"""
    return all(x is not None for x in xs)


def _nanmin(a):
    return float(np.nanmin(a)) if np.any(~np.isnan(a)) else None


def _nanmax(a):
    return float(np.nanmax(a)) if np.any(~np.isnan(a)) else None


# =============================================================================
# 5) FM-6 / FM-9 — Omuz fleksiyonu
# =============================================================================

class _ShoulderFlexionBase(BaseAnalyzer):
    section = "III"
    max_score = 2
    window_frac = 0.2
    onset_check = True
    min_flex_for_score = 30.0
    full_flex = 65.0

    def analyze(self, frames, side, fps):
        s = series_arm_angles(frames, side)
        T = len(frames)
        ref_idx = max(2, T // 10)
        ref_len = float(np.nanmedian(s["upper_arm_len"][:ref_idx]))
        if not np.isfinite(ref_len) or ref_len <= 0:
            ref_len = float(np.nanmedian(s["upper_arm_len"]))

        flex_series = np.full(T, np.nan)
        for t in range(T):
            sh, el = s["shoulder_xy"][t], s["elbow_xy"][t]
            if not np.isnan(sh).any() and not np.isnan(el).any():
                flex_series[t] = _shoulder_flexion_proxy_2d(sh, el, ref_len)

        abd_max = _nanmax(s["shoulder_abd"]) or 0.0
        max_flex = max(_nanmax(flex_series) or 0.0, abd_max)

        i0, i1 = _peak_motion_window(flex_series, self.window_frac)
        elbow_during = s["elbow_joint"][i0:i1]
        elbow_drop = 50.0
        elbow_comp = bool(np.any(~np.isnan(elbow_during))
                          and np.nanmin(elbow_during) < (180.0 - elbow_drop))

        elbow_imm = False
        if self.onset_check:
            first_end = i0 + max(1, int(0.25 * (i1 - i0)))
            seg = s["elbow_joint"][i0:first_end]
            elbow_imm = bool(np.any(~np.isnan(seg))
                             and np.nanmin(seg) < (180.0 - elbow_drop))

        score = self._score(max_flex, elbow_comp, elbow_imm)
        notes = [f"max sagital flex ≈ {max_flex:.0f}°"]
        if elbow_comp:
            notes.append("dirsek bükülmesiyle (kompansasyon)")
        return self._result(score, side,
                            measurements={
                                "max_shoulder_flexion_deg": round(max_flex, 1),
                                "min_elbow_during_deg": round(_nanmin(elbow_during), 1)
                                    if _nanmin(elbow_during) is not None else None,
                                "upper_arm_ref_len_px": round(ref_len, 1)},
                            notes=notes)

    def _score(self, max_flex, elbow_comp, elbow_imm):
        raise NotImplementedError


class FM6_ShoulderFlexion0to90(_ShoulderFlexionBase):
    test_id = "FM-6"
    test_name = "Shoulder flexion 0°-90° (elbow 0°, neutral forearm)"
    section = "III"
    window_frac = 0.2

    def _score(self, max_flex, elbow_comp, elbow_imm):
        if max_flex < 30.0 or elbow_imm:
            return 0
        if max_flex >= 65.0 and not elbow_comp:
            return 2
        return 1


class FM9_ShoulderFlexion90to180(_ShoulderFlexionBase):
    test_id = "FM-9"
    test_name = "Shoulder flexion 90°-180° (elbow 0°)"
    section = "IV"
    window_frac = 0.25
    onset_check = False

    def _score(self, max_flex, elbow_comp, elbow_imm):
        if max_flex < 95.0:
            return 0
        if max_flex >= 150.0 and not elbow_comp:
            return 2
        if elbow_comp:
            return 1 if max_flex >= 110 else 0
        return 1


# =============================================================================
# 6) FM-8 — Omuz abduksiyonu 0°-90°
# =============================================================================

class FM8_ShoulderAbduction0to90(BaseAnalyzer):
    test_id = "FM-8"
    test_name = "Shoulder abduction 0°-90° (elbow 0°, forearm neutral)"
    section = "IV"
    max_score = 2

    def analyze(self, frames, side, fps):
        s = series_arm_angles(frames, side)
        max_abd = _nanmax(s["shoulder_abd"]) or 0.0
        i0, i1 = _peak_motion_window(s["shoulder_abd"], 0.25)
        elbow_during = s["elbow_joint"][i0:i1]
        rot_during = s["forearm_rot"][i0:i1]

        elbow_drop = 50.0
        sup_thresh = 45.0   # forearm_rot ∈ [-90,90]; ulaşılabilir eşik (bug fix)
        elbow_comp = bool(np.any(~np.isnan(elbow_during))
                          and np.nanmin(elbow_during) < (180.0 - elbow_drop))
        sup_comp = bool(np.any(~np.isnan(rot_during))
                        and np.nanmax(rot_during) > sup_thresh)

        first_end = i0 + max(1, int(0.25 * (i1 - i0)))
        seg = s["elbow_joint"][i0:first_end]
        elbow_imm = bool(np.any(~np.isnan(seg))
                         and np.nanmin(seg) < (180.0 - elbow_drop))

        if max_abd < 25.0 or elbow_imm:
            score = 0
        elif max_abd >= 70.0 and not (elbow_comp or sup_comp):
            score = 2
        else:
            score = 1

        notes = [f"max abd ≈ {max_abd:.0f}°"]
        if elbow_comp:
            notes.append("dirsek fleksiyonu sırasında")
        if sup_comp:
            notes.append("supinasyon sırasında")
        return self._result(score, side,
                            measurements={
                                "max_abduction_deg": round(max_abd, 1),
                                "min_elbow_during_deg": round(_nanmin(elbow_during), 1)
                                    if _nanmin(elbow_during) is not None else None,
                                "max_forearm_rot_deg": round(_nanmax(rot_during), 1)
                                    if _nanmax(rot_during) is not None else None},
                            notes=notes)


# =============================================================================
# 7) FM-7 / FM-10 — Pronasyon / Supinasyon
# =============================================================================

class _PronSupBase(BaseAnalyzer):
    section = "III"
    max_score = 2
    expected_elbow_low = 60.0
    expected_elbow_high = 120.0
    expected_abd_low = 0.0
    expected_abd_high = 30.0

    def analyze(self, frames, side, fps):
        s = series_arm_angles(frames, side)
        rot = s["forearm_rot"]
        rng = (_nanmax(rot) - _nanmin(rot)) if _nanmax(rot) is not None else 0.0
        elbow_med = float(np.nanmedian(s["elbow_joint"])) \
            if np.any(~np.isnan(s["elbow_joint"])) else 180.0
        abd_med = float(np.nanmedian(s["shoulder_abd"])) \
            if np.any(~np.isnan(s["shoulder_abd"])) else 0.0
        posture_ok = (self.expected_elbow_low <= elbow_med <= self.expected_elbow_high
                      and self.expected_abd_low <= abd_med <= self.expected_abd_high)

        score = 2 if rng >= 60.0 else 1 if rng >= 25.0 else 0
        notes = [f"rotation range ≈ {rng:.0f}°",
                 f"elbow_med={elbow_med:.0f}° abd_med={abd_med:.0f}°"]
        if not posture_ok:
            notes.append("başlangıç pozu sapmış (düşük puan riski)")
        return self._result(score, side,
                            measurements={
                                "forearm_rot_range_deg": round(rng, 1),
                                "forearm_rot_min_deg": round(_nanmin(rot), 1)
                                    if _nanmin(rot) is not None else None,
                                "forearm_rot_max_deg": round(_nanmax(rot), 1)
                                    if _nanmax(rot) is not None else None,
                                "elbow_median_deg": round(elbow_med, 1),
                                "abd_median_deg": round(abd_med, 1),
                                "posture_ok": posture_ok},
                            notes=notes)


class FM7_PronSup_Elbow90(_PronSupBase):
    test_id = "FM-7"
    test_name = "Pronation/Supination (elbow 90°, shoulder 0°)"
    section = "III"
    expected_elbow_low, expected_elbow_high = 60.0, 120.0
    expected_abd_low, expected_abd_high = 0.0, 30.0


class FM10_PronSup_Elbow0(_PronSupBase):
    test_id = "FM-10"
    test_name = "Pronation/Supination (elbow 0°, shoulder ~30° flexion)"
    section = "IV"
    expected_elbow_low, expected_elbow_high = 150.0, 180.0
    expected_abd_low, expected_abd_high = 0.0, 50.0


# =============================================================================
# 8) FM-11 / FM-12 — El bileği stabilite + tekrarlı F/E
# =============================================================================

class _WristMotionBase(BaseAnalyzer):
    section = "B"
    max_score = 4
    expected_elbow_low = 0.0
    expected_elbow_high = 180.0

    def analyze(self, frames, side, fps):
        s = series_arm_angles(frames, side)
        wf = s["wrist_flex"]
        elbow_med = float(np.nanmedian(s["elbow_joint"])) \
            if np.any(~np.isnan(s["elbow_joint"])) else 180.0
        posture_ok = self.expected_elbow_low <= elbow_med <= self.expected_elbow_high

        if np.all(np.isnan(wf)):
            stability_score = range_score = 0
            wf_max = wf_min = wf_std = 0.0
            cycles = 0
        else:
            wf_max = float(np.nanmax(wf))
            wf_min = float(np.nanmin(wf))
            peak_idx = int(np.nanargmax(wf))
            half = max(2, len(wf) // 20)
            i0, i1 = max(0, peak_idx - half), min(len(wf), peak_idx + half)
            wf_std = float(np.nanstd(wf[i0:i1])) if i1 > i0 else 0.0
            if abs(wf_max) < 8.0:
                stability_score = 0
            elif abs(wf_max) < 15.0:
                stability_score = 1
            else:
                stability_score = 2 if wf_std < 22.0 else 1
            wf_range = wf_max - wf_min
            cycles = self._count_cycles(wf)
            smooth = self._smoothness_ratio(wf)
            if wf_range < 15.0 or cycles < 1:
                range_score = 0
            elif wf_range >= 30.0 and cycles >= 2:
                range_score = 2 if smooth < 0.30 else 1
            else:
                range_score = 1

        if not posture_ok:
            stability_score = max(0, stability_score - 1)
            range_score = max(0, range_score - 1)

        total = stability_score + range_score
        notes = [f"wrist range = {wf_max - wf_min:.0f}° (peak {wf_max:.0f}°)",
                 f"cycles ≈ {cycles}", f"elbow_med = {elbow_med:.0f}°"]
        if not posture_ok:
            notes.append("başlangıç pozu sapmış")
        return self._result(total, side,
                            sub_scores={"stability": stability_score,
                                        "repeated_fe": range_score},
                            measurements={
                                "wrist_max_deg": round(wf_max, 1),
                                "wrist_min_deg": round(wf_min, 1),
                                "wrist_range_deg": round(wf_max - wf_min, 1),
                                "wrist_peak_std_deg": round(wf_std, 1),
                                "wrist_smoothness": round(self._smoothness_ratio(wf), 3),
                                "cycles_estimate": cycles,
                                "elbow_median_deg": round(elbow_med, 1),
                                "posture_ok": posture_ok},
                            notes=notes)

    @staticmethod
    def _count_cycles(signal: np.ndarray) -> int:
        s = signal[~np.isnan(signal)]
        if len(s) < 4:
            return 0
        s = s - np.median(s)
        zc = np.where(np.diff(np.signbit(s)))[0]
        return len(zc) // 2

    @staticmethod
    def _smoothness_ratio(signal: np.ndarray) -> float:
        s = signal[~np.isnan(signal)]
        if len(s) < 5:
            return 1.0
        d2 = np.diff(s, n=2)
        if len(d2) == 0:
            return 1.0
        return float(np.std(d2) / (np.ptp(s) + 1e-6))


class FM11_WristAt90(_WristMotionBase):
    test_id = "FM-11"
    test_name = "Wrist stability + repeated F/E (elbow 90°, pronated)"
    expected_elbow_low, expected_elbow_high = 60.0, 120.0


class FM12_WristAt0(_WristMotionBase):
    test_id = "FM-12"
    test_name = "Wrist stability + repeated F/E (elbow 0°)"
    expected_elbow_low, expected_elbow_high = 150.0, 180.0


# =============================================================================
# 9) FM-13 — El bileği sirkümdüksiyonu
# =============================================================================

class FM13_WristCircumduction(BaseAnalyzer):
    test_id = "FM-13"
    test_name = "Wrist circumduction (elbow 90°, pronated, shoulder 0°)"
    section = "B"
    max_score = 2

    def analyze(self, frames, side, fps):
        s = _side_idx(side)
        traj = []
        for f in frames:
            mc = f.hand_xy(s["hand"], HAND_MIDDLE_MCP)
            if mc is not None:
                traj.append(mc)
        if len(traj) < 20:
            return self._result(0, side,
                                measurements={"trajectory_points": len(traj)},
                                notes=["yeterli el merkez tespiti yok"])
        traj = np.array(traj)
        rel = traj - traj.mean(0)
        cov = np.cov(rel.T)
        eig = np.sort(np.linalg.eigvalsh(cov))
        circular_ratio = float(eig[0] / (eig[1] + 1e-9))
        radii = np.linalg.norm(rel, axis=1)
        r_med = float(np.median(radii))
        path_len = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
        completion = path_len / (2 * np.pi * r_med + 1e-6)
        jerk = float(np.linalg.norm(np.diff(traj, axis=0, n=2), axis=1).mean())

        if r_med < 8.0:
            score = 0
        elif circular_ratio < 0.08 or completion < 0.6:
            score = 1
        elif circular_ratio >= 0.13 and completion >= 0.8 and jerk < 25.0:
            score = 2
        else:
            score = 1
        return self._result(score, side,
                            measurements={
                                "circular_ratio": round(circular_ratio, 3),
                                "completion_ratio": round(completion, 2),
                                "median_radius_px": round(r_med, 1),
                                "jerk_mean_px": round(jerk, 2),
                                "trajectory_points": len(traj)},
                            notes=[f"circular_ratio={circular_ratio:.2f}",
                                   f"completion={completion:.2f}×"])


# =============================================================================
# 10) FM-D — Koordinasyon / Hız (tremor + dismetri + süre)
# =============================================================================

class CoordinationSpeed(BaseAnalyzer):
    test_id = "FM-D"
    test_name = "Coordination / Speed — finger-to-nose 5x"
    section = "D"
    max_score = 6

    def _metrics(self, frames, side, fps) -> dict:
        s = _side_idx(side)
        T = len(frames)
        tip_xy = np.full((T, 2), np.nan)
        nose_xy = np.full((T, 2), np.nan)
        for t, f in enumerate(frames):
            tip = f.hand_xy(s["hand"], HAND_INDEX_TIP)
            ns = f.body_xy(NOSE, 0.4)
            if tip is not None:
                tip_xy[t] = tip
            if ns is not None:
                nose_xy[t] = ns
        d_nose = np.linalg.norm(tip_xy - nose_xy, axis=1)
        mask = ~np.isnan(d_nose)
        if mask.sum() < 30:
            return {"valid": False, "valid_frames": int(mask.sum())}
        idx = np.arange(T)
        d_fill = np.interp(idx, idx[mask], d_nose[mask])
        in_min = d_fill < np.percentile(d_fill, 25)
        groups, cur = [], []
        for i, b in enumerate(in_min):
            if b:
                cur.append(i)
            elif cur:
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
        touches = [int(np.mean(g)) for g in groups if len(g) >= 2]
        n_cycles = len(touches)
        if n_cycles >= 2:
            dur = (touches[-1] - touches[0]) / fps
            full_dur = dur * (5 / max(1, n_cycles - 1))
        else:
            full_dur = float("nan")

        from numpy.fft import rfft, rfftfreq
        x_fill = np.interp(idx, idx[mask], tip_xy[:, 0][mask])
        y_fill = np.interp(idx, idx[mask], tip_xy[:, 1][mask])
        x_d = x_fill - np.convolve(x_fill, np.ones(7) / 7, mode="same")
        y_d = y_fill - np.convolve(y_fill, np.ones(7) / 7, mode="same")
        Fx = np.abs(rfft(x_d)) / max(T, 1)
        Fy = np.abs(rfft(y_d)) / max(T, 1)
        freqs = rfftfreq(T, d=1 / fps)
        hf = (freqs > 3.0) & (freqs < 12.0)
        tremor_raw = float((Fx[hf].mean() + Fy[hf].mean()) / 2.0) if hf.any() else 0.0

        arm_len = []
        for f in frames:
            sh = f.body_xy(s["shoulder"])
            wr = f.body_xy(s["wrist"])
            if sh is not None and wr is not None:
                arm_len.append(float(np.linalg.norm(wr - sh)))
        scale = float(np.median(arm_len)) if arm_len else 100.0
        tremor = tremor_raw / max(scale, 1.0) * 100.0

        if len(touches) >= 2:
            md = np.array(sorted([d_fill[i] for i in touches])[:5])
            mad = float(np.median(np.abs(md - np.median(md))))
            scatter = mad * 1.4826 / max(scale, 1.0) * 100.0
        else:
            scatter = float("nan")
        return {"valid": True, "n_cycles": n_cycles, "duration_5x": full_dur,
                "tremor": tremor, "scatter": scatter, "arm_scale_px": scale}

    def analyze(self, frames, side, fps):
        m = self._metrics(frames, side, fps)
        if not m["valid"]:
            return self._result(0, side,
                                measurements={"valid_frames": m.get("valid_frames", 0)},
                                notes=["yetersiz tespit"])
        fd = m["duration_5x"]
        time_score = 0 if not np.isfinite(fd) else 2 if fd < 12 else 1 if fd < 20 else 0
        tr = m["tremor"]
        tremor_score = 0 if tr > 5 else 1 if tr > 2 else 2
        sc = m["scatter"]
        dys_score = 0 if not np.isfinite(sc) else 0 if sc > 50 else 1 if sc > 25 else 2
        total = tremor_score + dys_score + time_score
        return self._result(total, side,
                            sub_scores={"tremor": tremor_score, "dysmetria": dys_score,
                                        "time": time_score},
                            measurements={
                                "n_cycles": m["n_cycles"],
                                "duration_5x_seconds": round(fd, 1) if np.isfinite(fd) else None,
                                "tremor_energy": round(tr, 2),
                                "dysmetria_scatter_pct": round(sc, 1) if np.isfinite(sc) else None},
                            notes=["(referans yok — mutlak eşikle puanlandı)"])

    def analyze_paired(self, fa, sa, fu, su, fps) -> FMTestResult:
        ma = self._metrics(fa, sa, fps)
        mu = self._metrics(fu, su, fps)
        if not ma["valid"] or not mu["valid"]:
            bad = "etkilenen" if not ma["valid"] else "etkilenmemiş"
            return self._result(0, sa, measurements={"error": f"{bad} tarafta yetersiz tespit"},
                                notes=["paired analiz başarısız"])
        da, du = ma["duration_5x"], mu["duration_5x"]
        if np.isfinite(da) and np.isfinite(du):
            dd = max(0.0, da - du)
            time_score = 2 if dd < 6 else 1 if dd < 10 else 0
        else:
            dd = float("nan")
            time_score = 0
        tr = ma["tremor"] / max(mu["tremor"], 0.5)
        tremor_score = 2 if tr < 1.5 else 1 if tr < 3 else 0
        sa_, su_ = ma["scatter"], mu["scatter"]
        if not np.isfinite(sa_):
            dys_score, sr = 0, float("nan")
        else:
            sr = sa_ / (max(su_, 0.3) if np.isfinite(su_) else 0.3)
            dys_score = 2 if (sa_ < 5 or sr < 1.5) else 1 if sr < 3 else 0
        total = tremor_score + dys_score + time_score
        return self._result(total, sa,
                            sub_scores={"tremor": tremor_score, "dysmetria": dys_score,
                                        "time": time_score},
                            measurements={
                                "duration_aff_s": round(da, 1) if np.isfinite(da) else None,
                                "duration_unaff_s": round(du, 1) if np.isfinite(du) else None,
                                "duration_diff_s": round(dd, 1) if np.isfinite(dd) else None,
                                "tremor_ratio": round(tr, 2),
                                "dysmetria_ratio": round(sr, 2) if np.isfinite(sr) else None,
                                "n_cycles_aff": ma["n_cycles"],
                                "n_cycles_unaff": mu["n_cycles"]},
                            notes=["(resmi FMA-D: etkilenmemiş taraf referans)"])


# =============================================================================
# 11) BÖLÜM II — FLEKSÖR & EKSTANSÖR SİNERJİSİ
# =============================================================================
#
# Puanlama, farklı karelerin bağımsız peak'lerini birleştirmek yerine TEK bir
# eşzamanlı "bitiş pozu" karesinden yapılır (skor şişmesini önler). Bitiş pozu,
# etkilenen kolun en abdükte/adükte olduğu kare olarak seçilir; el-bağımlı
# rotasyon ölçümü o kare civarındaki kısa pencerede en iyi değerle desteklenir.


@dataclass
class FlexorSynergyResult:
    glenohumeral_abduction_score: int = 0
    elbow_flexion_score: int = 0
    forearm_supination_score: int = 0
    gh_abduction_deg: float = 0.0
    gh_abduction_h_deg: float = 0.0
    elbow_flex_deg: float = 0.0
    elbow_flex_h_deg: float = 0.0
    supination_deg: float = 0.0
    supination_h_deg: float = 0.0
    elevation_diff_px: float = 0.0

    @property
    def total(self) -> int:
        return (self.glenohumeral_abduction_score + self.elbow_flexion_score
                + self.forearm_supination_score)

    max_score = 6

    def to_dict(self) -> dict:
        return {
            "test": "flexor_synergy",
            "subtests": {
                "glenohumeral_abduction": {"score": self.glenohumeral_abduction_score,
                                           "angle_affected_deg": round(self.gh_abduction_deg, 1),
                                           "angle_healthy_deg": round(self.gh_abduction_h_deg, 1)},
                "elbow_flexion": {"score": self.elbow_flexion_score,
                                  "angle_affected_deg": round(self.elbow_flex_deg, 1),
                                  "angle_healthy_deg": round(self.elbow_flex_h_deg, 1)},
                "forearm_supination": {"score": self.forearm_supination_score,
                                       "rotation_deg": round(self.supination_deg, 1),
                                       "rotation_healthy_deg": round(self.supination_h_deg, 1)},
            },
            "total": self.total, "max_score": self.max_score,
            "u_items_omitted": ["shoulder_retraction", "shoulder_elevation",
                                "shoulder_external_rotation"],
        }


class FlexorSynergyEvaluator:
    def __init__(self, affected_side="left", abduction_min_deg=90.0,
                 abduction_partial_deg=30.0, symmetry_margin_deg=15.0,
                 supination_threshold_deg=45.0):
        self.affected_side = affected_side
        self.abduction_min_deg = abduction_min_deg
        self.abduction_partial_deg = abduction_partial_deg
        self.symmetry_margin_deg = symmetry_margin_deg
        self.supination_threshold = supination_threshold_deg
        self._a = _side_idx(affected_side)
        self._h = _side_idx("right" if affected_side == "left" else "left")

    def _measure(self, frame: PoseFrame) -> dict:
        a, h = self._a, self._h
        out = {}
        a_sh = frame.body_xy(a["shoulder"]); a_el = frame.body_xy(a["elbow"])
        a_wr = frame.body_xy(a["wrist"]); a_hp = frame.body_xy(a["hip"])
        h_sh = frame.body_xy(h["shoulder"]); h_el = frame.body_xy(h["elbow"])
        h_wr = frame.body_xy(h["wrist"]); h_hp = frame.body_xy(h["hip"])
        a_hw = frame.hand_xy(a["hand"], HAND_WRIST)
        a_im = frame.hand_xy(a["hand"], HAND_INDEX_MCP)
        a_mm = frame.hand_xy(a["hand"], HAND_MIDDLE_MCP)
        h_hw = frame.hand_xy(h["hand"], HAND_WRIST)
        h_im = frame.hand_xy(h["hand"], HAND_INDEX_MCP)
        h_mm = frame.hand_xy(h["hand"], HAND_MIDDLE_MCP)

        out["gh"] = (shoulder_abduction_angle(a_sh, a_el, a_hp)
                     if _have(a_sh, a_el, a_hp) else None)
        out["gh_h"] = (shoulder_abduction_angle(h_sh, h_el, h_hp)
                       if _have(h_sh, h_el, h_hp) else None)
        out["elb"] = (elbow_flexion_amount(a_sh, a_el, a_wr)
                      if _have(a_sh, a_el, a_wr) else None)
        out["elb_h"] = (elbow_flexion_amount(h_sh, h_el, h_wr)
                        if _have(h_sh, h_el, h_wr) else None)
        out["sup"] = (forearm_rotation_angle(a_hw, a_im, a_mm)
                      if _have(a_hw, a_im, a_mm) else None)
        out["sup_h"] = (forearm_rotation_angle(h_hw, h_im, h_mm)
                        if _have(h_hw, h_im, h_mm) else None)
        if a_sh is not None and h_sh is not None:
            ls = a_sh if self.affected_side == "left" else h_sh
            rs = h_sh if self.affected_side == "left" else a_sh
            out["elev"] = shoulder_elevation_diff_px(ls, rs, self.affected_side)
        else:
            out["elev"] = None
        return out

    def evaluate_sequence(self, frames: list[PoseFrame]) -> FlexorSynergyResult:
        """Bitiş pozu = abduksiyonun en yüksek olduğu kare; puan o kareden.
        El-bağımlı supinasyon, abduksiyon penceresindeki en iyi değerle desteklenir."""
        T = len(frames)
        measures = [self._measure(f) for f in frames]
        gh = np.array([m["gh"] if m["gh"] is not None else np.nan for m in measures])

        best = FlexorSynergyResult()
        if not np.any(~np.isnan(gh)):
            return best
        end_idx = int(np.nanargmax(gh))
        m = measures[end_idx]

        # Bitiş pozu civarındaki pencere (el ölçümü gürültülü olabilir)
        half = max(2, T // 20)
        w0, w1 = max(0, end_idx - half), min(T, end_idx + half + 1)
        sup_win = [measures[i]["sup"] for i in range(w0, w1) if measures[i]["sup"] is not None]
        sup_h_win = [measures[i]["sup_h"] for i in range(w0, w1) if measures[i]["sup_h"] is not None]

        best.gh_abduction_deg = m["gh"] or 0.0
        best.gh_abduction_h_deg = m["gh_h"] or 0.0
        best.elbow_flex_deg = m["elb"] or 0.0
        best.elbow_flex_h_deg = m["elb_h"] or 0.0
        best.supination_deg = max(sup_win) if sup_win else (m["sup"] or 0.0)
        best.supination_h_deg = max(sup_h_win) if sup_h_win else (m["sup_h"] or 0.0)
        best.elevation_diff_px = m["elev"] or 0.0

        deg = best.gh_abduction_deg
        best.glenohumeral_abduction_score = (
            2 if deg >= self.abduction_min_deg
            else 1 if deg >= self.abduction_partial_deg else 0)
        best.elbow_flexion_score = (
            2 if best.elbow_flex_deg >= 110 else 1 if best.elbow_flex_deg >= 30 else 0)

        if best.supination_h_deg > 0 and best.supination_deg > 0:
            best.forearm_supination_score = symmetry_score(
                best.supination_deg, best.supination_h_deg, self.symmetry_margin_deg)
        elif best.supination_deg >= self.supination_threshold:
            best.forearm_supination_score = 2
        elif best.supination_deg >= 15.0:
            best.forearm_supination_score = 1
        elif best.glenohumeral_abduction_score == 2 and best.elbow_flex_deg >= 110:
            best.forearm_supination_score = 2  # el yok ama bitiş pozu doğru
        else:
            best.forearm_supination_score = 0
        return best


@dataclass
class ExtensorSynergyResult:
    shoulder_adduction_ir_score: int = 0
    elbow_extension_score: int = 0
    forearm_pronation_score: int = 0
    elbow_extension_deg: float = 0.0
    pronation_deg: float = 0.0
    crossed_midline: bool = False
    wrist_x: float = 0.0
    midline_x: float = 0.0
    midline_approach: float = 0.0

    @property
    def total(self) -> int:
        return (self.shoulder_adduction_ir_score + self.elbow_extension_score
                + self.forearm_pronation_score)

    max_score = 6

    def to_dict(self) -> dict:
        return {
            "test": "extensor_synergy",
            "subtests": {
                "shoulder_adduction_internal_rotation": {
                    "score": self.shoulder_adduction_ir_score,
                    "crossed_midline": self.crossed_midline,
                    "midline_approach_ratio": round(self.midline_approach, 2)},
                "elbow_extension": {"score": self.elbow_extension_score,
                                    "extension_deficit_deg": round(self.elbow_extension_deg, 1)},
                "forearm_pronation": {"score": self.forearm_pronation_score,
                                      "rotation_deg": round(self.pronation_deg, 1)},
            },
            "total": self.total, "max_score": self.max_score,
        }


class ExtensorSynergyEvaluator:
    def __init__(self, affected_side="left", extension_full_deg=10.0,
                 extension_partial_deg=90.0, pronation_full_deg=-45.0,
                 pronation_partial_deg=-20.0, midline_partial_ratio=0.5):
        self.affected_side = affected_side
        self.extension_full_deg = extension_full_deg
        self.extension_partial_deg = extension_partial_deg
        self.pronation_full_deg = pronation_full_deg
        self.pronation_partial_deg = pronation_partial_deg
        self.midline_partial_ratio = midline_partial_ratio
        self._a = _side_idx(affected_side)

    def _measure(self, frame: PoseFrame) -> dict:
        a = self._a
        a_sh = frame.body_xy(a["shoulder"]); a_el = frame.body_xy(a["elbow"])
        a_wr = frame.body_xy(a["wrist"])
        l_hp = frame.body_xy(LEFT_HIP); r_hp = frame.body_xy(RIGHT_HIP)
        a_hw = frame.hand_xy(a["hand"], HAND_WRIST)
        a_im = frame.hand_xy(a["hand"], HAND_INDEX_MCP)
        a_mm = frame.hand_xy(a["hand"], HAND_MIDDLE_MCP)
        out = {"approach": None, "midline_x": None, "wrist_x": None,
               "deficit": None, "pron": None}
        if _have(a_wr, a_sh, l_hp, r_hp):
            out["approach"] = midline_approach_ratio(a_wr, a_sh, l_hp, r_hp)
            out["midline_x"] = float((l_hp[0] + r_hp[0]) / 2.0)
            out["wrist_x"] = float(a_wr[0])
        if _have(a_sh, a_el, a_wr):
            out["deficit"] = 180.0 - joint_angle(a_sh, a_el, a_wr)
        if _have(a_hw, a_im, a_mm):
            out["pron"] = forearm_rotation_angle(a_hw, a_im, a_mm)
        return out

    def evaluate_sequence(self, frames: list[PoseFrame]) -> ExtensorSynergyResult:
        """Bitiş pozu = orta hatta en çok yaklaşıldığı kare; tüm alt-puanlar
        o kareden (eşzamanlı). Pronasyon o kare civarındaki en iyi değerle."""
        T = len(frames)
        measures = [self._measure(f) for f in frames]
        appr = np.array([m["approach"] if m["approach"] is not None else np.nan
                         for m in measures])
        best = ExtensorSynergyResult()
        if not np.any(~np.isnan(appr)):
            return best
        end_idx = int(np.nanargmax(appr))
        m = measures[end_idx]

        half = max(2, T // 20)
        w0, w1 = max(0, end_idx - half), min(T, end_idx + half + 1)
        pron_win = [measures[i]["pron"] for i in range(w0, w1)
                    if measures[i]["pron"] is not None]

        best.midline_approach = m["approach"] or 0.0
        best.midline_x = m["midline_x"] or 0.0
        best.wrist_x = m["wrist_x"] or 0.0
        # Taraf-duyarlı: orta hattı geçmek = yaklaşma oranı ≥ 1 (sabit < kaldırıldı)
        best.crossed_midline = best.midline_approach >= 1.0
        if best.crossed_midline:
            best.shoulder_adduction_ir_score = 2
        elif best.midline_approach >= self.midline_partial_ratio:
            best.shoulder_adduction_ir_score = 1
        else:
            best.shoulder_adduction_ir_score = 0

        if m["deficit"] is not None:
            best.elbow_extension_deg = m["deficit"]
            best.elbow_extension_score = (
                2 if m["deficit"] <= self.extension_full_deg
                else 1 if m["deficit"] <= self.extension_partial_deg else 0)

        best.pronation_deg = min(pron_win) if pron_win else (m["pron"] or 0.0)
        best.forearm_pronation_score = (
            2 if best.pronation_deg <= self.pronation_full_deg
            else 1 if best.pronation_deg <= self.pronation_partial_deg else 0)
        return best


# =============================================================================
# 12) REGISTRY + ÜST DÜZEY API
# =============================================================================

ANALYZERS: dict[str, BaseAnalyzer] = {
    "FM-6": FM6_ShoulderFlexion0to90(),
    "FM-7": FM7_PronSup_Elbow90(),
    "FM-8": FM8_ShoulderAbduction0to90(),
    "FM-9": FM9_ShoulderFlexion90to180(),
    "FM-10": FM10_PronSup_Elbow0(),
    "FM-11": FM11_WristAt90(),
    "FM-12": FM12_WristAt0(),
    "FM-13": FM13_WristCircumduction(),
    "FM-D": CoordinationSpeed(),
}

_FRIENDLY = {
    "fm6": "FM-6", "fm7": "FM-7", "fm8": "FM-8", "fm9": "FM-9", "fm10": "FM-10",
    "fm11": "FM-11", "fm12": "FM-12", "fm13": "FM-13", "fmd": "FM-D",
}


def _norm_test(test: str) -> str:
    t = (test or "all").strip().lower().replace("_", "-").replace(" ", "")
    if t in ("flexor", "flexor-synergy", "fleksor"):
        return "FLEXOR"
    if t in ("extensor", "extensor-synergy", "ekstansor"):
        return "EXTENSOR"
    if t in ("all", "section2", "section-ii", "synergy", "sinerji"):
        return "ALL"
    return _FRIENDLY.get(t.replace("-", ""), test.upper())


def _flexor_result(r: FlexorSynergyResult, side: str) -> dict:
    return FMTestResult("FLEXOR_SYNERGY", "Flexor synergy (Section II)", "II",
                        r.total, r.max_score, side,
                        sub_scores={"glenohumeral_abduction": r.glenohumeral_abduction_score,
                                    "elbow_flexion": r.elbow_flexion_score,
                                    "forearm_supination": r.forearm_supination_score},
                        measurements={"gh_abduction_deg": round(r.gh_abduction_deg, 1),
                                      "elbow_flex_deg": round(r.elbow_flex_deg, 1),
                                      "supination_deg": round(r.supination_deg, 1),
                                      "elevation_diff_px": round(r.elevation_diff_px, 1)}).to_dict()


def _extensor_result(r: ExtensorSynergyResult, side: str) -> dict:
    return FMTestResult("EXTENSOR_SYNERGY", "Extensor synergy (Section II)", "II",
                        r.total, r.max_score, side,
                        sub_scores={"shoulder_adduction_ir": r.shoulder_adduction_ir_score,
                                    "elbow_extension": r.elbow_extension_score,
                                    "forearm_pronation": r.forearm_pronation_score},
                        measurements={"elbow_extension_deficit_deg": round(r.elbow_extension_deg, 1),
                                      "pronation_deg": round(r.pronation_deg, 1),
                                      "crossed_midline": r.crossed_midline,
                                      "midline_approach": round(r.midline_approach, 2)}).to_dict()


def analyze(v: str, *, test: str = "all", side: str = "auto",
            model: str = "heavy", unaffected: Optional[str] = None) -> dict:
    """Fugl-Meyer üst ekstremite analizi.

    v          : pose+el CSV yolu (route'tan) veya video yolu.
    test       : flexor | extensor | all | FM-6..FM-13 | FM-D
    side       : auto | left | right (etkilenen taraf)
    unaffected : FM-D paired skorlama için etkilenmemiş taraf CSV'si (ops).
    """
    frames, fps = load_frames(v, model)
    valid = [f for f in frames if np.any(f.bvis > 0.3)]
    if len(valid) < 5:
        return {"error": "yetersiz pose tespiti (<5 geçerli kare)",
                "valid_frames": len(valid)}

    resolved = side if side in ("left", "right") else detect_active_side(valid)
    tid = _norm_test(test)

    if tid in ("FLEXOR", "ALL"):
        flx = FlexorSynergyEvaluator(affected_side=resolved).evaluate_sequence(frames)
    if tid in ("EXTENSOR", "ALL"):
        ext = ExtensorSynergyEvaluator(affected_side=resolved).evaluate_sequence(frames)

    if tid == "FLEXOR":
        return {"side": resolved, "fps": round(fps, 2), **_flexor_result(flx, resolved)}
    if tid == "EXTENSOR":
        return {"side": resolved, "fps": round(fps, 2), **_extensor_result(ext, resolved)}
    if tid == "ALL":
        total = flx.total + ext.total
        max_s = flx.max_score + ext.max_score
        return {
            "side": resolved, "fps": round(fps, 2), "section": "II",
            "total": total, "max_score": max_s,
            "percentage": round(total / max_s * 100.0, 1) if max_s else 0.0,
            "flexor_synergy": flx.to_dict(),
            "extensor_synergy": ext.to_dict(),
        }

    if tid == "FM-D" and unaffected:
        fu, _ = load_frames(unaffected, model)
        vu = [f for f in fu if np.any(f.bvis > 0.3)]
        side_u = detect_active_side(vu) if vu else ("right" if resolved == "left" else "left")
        res = CoordinationSpeed().analyze_paired(frames, resolved, fu, side_u, fps)
        return {"side": resolved, "fps": round(fps, 2), **res.to_dict()}

    analyzer = ANALYZERS.get(tid)
    if analyzer is None:
        return {"error": f"bilinmeyen test: {test}",
                "available": ["flexor", "extensor", "all", *sorted(ANALYZERS)]}
    res = analyzer.analyze(frames, resolved, fps)
    return {"side": resolved, "fps": round(fps, 2), **res.to_dict()}
