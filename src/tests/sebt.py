"""SEBT — Star Excursion Balance Test (yildiz uzanim denge testi).

Protokol: danisan tek bacak (stance) uzerinde durur, karsi ayagiyla
(reach) yildiz kollarina (yon) dogru maksimum uzanir; her uzanimin TEPE
karesi yakalanir. Tipik Y-Balance varyanti: anterior, posteromedial,
posterolateral; her iki bacak icin.

Girdi standart pose CSV'sinden FARKLIDIR: surekli akis degil, doktorun
sectigi tepe kareleridir. Her satir bir uzanim; basta `reach_id, leg,
direction` kolonlari vardir. Ayni uzanim on (front) ve yan (side) kamerada
ayni `reach_id` ile gelir.

Eksen yorumu:
  front.csv → goruntu X = mediolateral (sol/sag)  -> ML bileseni
  side.csv  → goruntu X = anteroposterior (on/arka) -> AP bileseni

Cikti: her reach icin reach_cm / normalize %; bacak basina kompozit;
yonlu simetri (anterior ham fark > 4 cm -> asymmetry_flag).

Detayli sozlesme: features/analysis/modules/sebt/BACKEND_SPEC.md
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import logging
import math
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..pose_common import (
    NOSE, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE, LEFT_FOOT, RIGHT_FOOT,
)

_log = logging.getLogger(__name__)

# Antropometri: burun->ayak bilegi dikey mesafe ~ boy * 0.93 (promptta belirtilen).
NOSE_ANKLE_RATIO = 0.93
VIS_THR = 0.3                         # foot_index gorunurluk esigi (yoksa ankle)
ASYMMETRY_ANTERIOR_CM = 4.0           # anterior ham fark esigi

SIDE_IDX = {
    "left":  {"hip": LEFT_HIP,  "ankle": LEFT_ANKLE,  "foot": LEFT_FOOT},
    "right": {"hip": RIGHT_HIP, "ankle": RIGHT_ANKLE, "foot": RIGHT_FOOT},
}


@dataclass
class _Rec:
    reach_id: int
    leg: str                  # stance bacak
    direction: str
    width: int
    height: int
    xy: np.ndarray            # (33, 2) normalize 0..1
    vis: np.ndarray           # (33,)
    world: Optional[np.ndarray]   # (33, 3) metre veya None


# ── CSV parse (SEBT semasi) ──────────────────────────────────────────────
def _parse(path: Optional[str]) -> dict[int, _Rec]:
    """SEBT CSV'sini reach_id -> _Rec haritasina cevirir. Ayni reach_id
    birden cok satirda gelirse sonuncu kazanir."""
    out: dict[int, _Rec] = {}
    if not path:
        return out
    with open(path, newline="") as fh:
        reader = _csv.DictReader(fh)
        if not reader.fieldnames:
            return out
        has_world = "wx0" in reader.fieldnames

        def num(row, name, default=None):
            v = row.get(name)
            if v is None or str(v).strip() == "":
                return default
            try:
                return float(v)
            except ValueError:
                return default

        for row in reader:
            rid = num(row, "reach_id")
            if rid is None:
                continue
            rid = int(rid)
            leg = (row.get("leg") or "").strip().lower()
            direction = (row.get("direction") or "").strip().lower()
            width = int(num(row, "width", 0) or 0)
            height = int(num(row, "height", 0) or 0)

            xy = np.full((33, 2), np.nan)
            vis = np.zeros(33)
            world = np.full((33, 3), np.nan) if has_world else None
            ok = True
            for i in range(33):
                xs, ys = num(row, f"x{i}"), num(row, f"y{i}")
                if xs is None or ys is None:
                    ok = False
                    break
                xy[i] = (xs, ys)
                vs = num(row, f"v{i}")
                vis[i] = vs if vs is not None else 1.0
                if has_world:
                    wx, wy, wz = (num(row, f"wx{i}"), num(row, f"wy{i}"),
                                  num(row, f"wz{i}"))
                    world[i] = (wx if wx is not None else np.nan,
                                wy if wy is not None else np.nan,
                                wz if wz is not None else np.nan)
            if not ok:
                continue
            out[rid] = _Rec(rid, leg, direction, width, height, xy, vis, world)
    return out


# ── Geometri yardimcilari ────────────────────────────────────────────────
def _foot_x(rec: _Rec, side: str) -> Optional[float]:
    """Verilen tarafin ayak normalize X'i: foot_index (gorunurse) yoksa ankle."""
    foot, ankle = SIDE_IDX[side]["foot"], SIDE_IDX[side]["ankle"]
    idx = foot if rec.vis[foot] >= VIS_THR else ankle
    x = float(rec.xy[idx][0])
    return x if np.isfinite(x) else None


def _px_per_cm(rec: _Rec, stance: str, height_m: Optional[float]) -> Optional[float]:
    """Kamera basina px/cm kalibrasyonu (burun->stance ayak bilegi dikey).

    World landmark varsa metre cinsinden gercek dikey mesafe; yoksa
    height_m * 0.93 antropometrik tahmini kullanilir."""
    ankle = SIDE_IDX[stance]["ankle"]
    ny, ay = float(rec.xy[NOSE][1]), float(rec.xy[ankle][1])
    if not (np.isfinite(ny) and np.isfinite(ay)):
        return None
    pix_vert = abs(ny - ay) * rec.height
    if pix_vert < 1.0:
        return None
    cm: Optional[float] = None
    if rec.world is not None:
        wny, way = float(rec.world[NOSE][1]), float(rec.world[ankle][1])
        if np.isfinite(wny) and np.isfinite(way):
            d = abs(wny - way) * 100.0
            if d > 1.0:
                cm = d
    if cm is None:
        if not height_m or height_m <= 0:
            return None
        cm = height_m * 100.0 * NOSE_ANKLE_RATIO
    return pix_vert / cm


def _limb_cm(fr: Optional[_Rec], sd: Optional[_Rec], stance: str,
             ppcf: Optional[float], ppcs: Optional[float]) -> Optional[float]:
    """Stance tarafi hip->ankle piksel mesafesinden bacak uzunlugu (cm).
    Iki kamera varsa ortalamasi."""
    ests: list[float] = []
    for rec, ppc in ((fr, ppcf), (sd, ppcs)):
        if rec is None or not ppc:
            continue
        hip = rec.xy[SIDE_IDX[stance]["hip"]]
        ankle = rec.xy[SIDE_IDX[stance]["ankle"]]
        if not (np.isfinite(hip).all() and np.isfinite(ankle).all()):
            continue
        pix = math.hypot((hip[0] - ankle[0]) * rec.width,
                         (hip[1] - ankle[1]) * rec.height)
        if pix > 1.0:
            ests.append(pix / ppc)
    return float(np.mean(ests)) if ests else None


def _dx_px(rec: Optional[_Rec], stance: str, reach_side: str) -> Optional[float]:
    """(reach_foot.x - stance_foot.x) * width — yatay uzanim bileseni (px)."""
    if rec is None or rec.width <= 0:
        return None
    rfoot, sfoot = _foot_x(rec, reach_side), _foot_x(rec, stance)
    if rfoot is None or sfoot is None:
        return None
    return (rfoot - sfoot) * rec.width


def _csv_list(s: Optional[str]) -> Optional[list[str]]:
    if not s:
        return None
    items = [x.strip().lower() for x in str(s).split(",") if x.strip()]
    return items or None


def _r(v, n=1):
    return round(float(v), n) if v is not None and np.isfinite(v) else None


# ── Ana analiz ───────────────────────────────────────────────────────────
def analyze(front_path: Optional[str] = None, side_path: Optional[str] = None,
            *, height_m: Optional[float] = None,
            directions: Optional[str] = None, legs: Optional[str] = None,
            limb_length_cm: Optional[float] = None) -> dict:
    front = _parse(front_path)
    side = _parse(side_path)
    if not front and not side:
        raise RuntimeError("front.csv veya side.csv gerekli (en az bir kamera).")

    both_cams = bool(front) and bool(side)
    all_ids = sorted(set(front) | set(side))
    if not all_ids:
        raise RuntimeError("Hic reach satiri yok.")

    ppc_front: list[float] = []
    ppc_side: list[float] = []
    limb_acc: dict[str, list[float]] = {"right": [], "left": []}
    rows: list[dict] = []

    for rid in all_ids:
        fr, sd = front.get(rid), side.get(rid)
        rec_any = fr or sd
        stance = (rec_any.leg or "").lower()
        if stance not in SIDE_IDX:
            _log.warning("reach_id %s gecersiz leg=%r -> atlandi", rid, stance)
            continue
        reach_side = "left" if stance == "right" else "right"
        direction = rec_any.direction

        if both_cams and (fr is None or sd is None):
            _log.warning("reach_id %s tek kamerada (eslesmeyen) -> tek eksenli", rid)

        ppcf = _px_per_cm(fr, stance, height_m) if fr else None
        ppcs = _px_per_cm(sd, stance, height_m) if sd else None
        if ppcf:
            ppc_front.append(ppcf)
        if ppcs:
            ppc_side.append(ppcs)

        dx_front = _dx_px(fr, stance, reach_side)   # ML (front)
        dx_side = _dx_px(sd, stance, reach_side)     # AP (side)

        comps = [c for c in (dx_front, dx_side) if c is not None]
        reach_px = math.sqrt(sum(c * c for c in comps)) if comps else None

        ppcs_avail = [p for p in (ppcf, ppcs) if p]
        ppc_mean = float(np.mean(ppcs_avail)) if ppcs_avail else None
        reach_cm = (reach_px / ppc_mean) if (reach_px is not None and ppc_mean) else None

        limb_est = _limb_cm(fr, sd, stance, ppcf, ppcs)
        if limb_est:
            limb_acc[stance].append(limb_est)

        rows.append({
            "reach_id": rid, "leg": stance, "direction": direction,
            "dx_front_px": dx_front, "dx_side_px": dx_side,
            "reach_px": reach_px, "reach_cm": reach_cm,
        })

    # ── Bacak uzunlugu (olculmus -> her iki bacak; yoksa pose tahmini) ──
    if limb_length_cm and limb_length_cm > 0:
        limb_length = {"right": float(limb_length_cm), "left": float(limb_length_cm)}
    else:
        limb_length = {
            leg: (float(np.mean(v)) if v else None) for leg, v in limb_acc.items()
        }

    # ── Normalize % ──────────────────────────────────────────────────────
    for r in rows:
        ll = limb_length.get(r["leg"])
        r["normalized_pct"] = (r["reach_cm"] / ll * 100.0
                               if (r["reach_cm"] is not None and ll) else None)

    # ── Kompozit (bacak basina normalize ortalama) ───────────────────────
    composite: dict[str, Optional[float]] = {}
    for leg in ("right", "left"):
        vals = [r["normalized_pct"] for r in rows
                if r["leg"] == leg and r["normalized_pct"] is not None]
        composite[leg] = float(np.mean(vals)) if vals else None

    # ── Simetri (yon basina sag-sol) ─────────────────────────────────────
    by_dir: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_dir.setdefault(r["direction"], {})[r["leg"]] = r

    worst_direction, worst_diff = None, -1.0
    for d, legs_map in by_dir.items():
        rr, ll = legs_map.get("right"), legs_map.get("left")
        if rr and ll and rr["normalized_pct"] is not None and ll["normalized_pct"] is not None:
            diff = abs(rr["normalized_pct"] - ll["normalized_pct"])
            if diff > worst_diff:
                worst_diff, worst_direction = diff, d

    ant = by_dir.get("anterior", {})
    ar, al = ant.get("right"), ant.get("left")
    anterior_diff_cm = None
    if ar and al and ar["reach_cm"] is not None and al["reach_cm"] is not None:
        anterior_diff_cm = abs(ar["reach_cm"] - al["reach_cm"])
    asymmetry_flag = bool(anterior_diff_cm is not None
                          and anterior_diff_cm > ASYMMETRY_ANTERIOR_CM)

    # ── Yon / bacak listeleri (form > pose'tan turetme) ──────────────────
    dirs_out = _csv_list(directions) or sorted({r["direction"] for r in rows})
    legs_out = _csv_list(legs) or sorted({r["leg"] for r in rows})

    return {
        "limb_length_cm": {k: _r(v, 1) for k, v in limb_length.items()},
        "px_per_cm": {
            "front": _r(np.mean(ppc_front), 2) if ppc_front else None,
            "side": _r(np.mean(ppc_side), 2) if ppc_side else None,
        },
        "reaches": [{
            "reach_id": r["reach_id"], "leg": r["leg"], "direction": r["direction"],
            "reach_cm": _r(r["reach_cm"], 1), "reach_px": _r(r["reach_px"], 1),
            "dx_front_px": _r(r["dx_front_px"], 1), "dx_side_px": _r(r["dx_side_px"], 1),
            "normalized_pct": _r(r["normalized_pct"], 1),
        } for r in rows],
        "composite": {k: _r(v, 1) for k, v in composite.items()},
        "symmetry": {
            "anterior_diff_cm": _r(anterior_diff_cm, 1),
            "asymmetry_flag": asymmetry_flag,
            "worst_direction": worst_direction,
        },
        "directions": dirs_out,
        "legs": legs_out,
    }


def main():
    ap = argparse.ArgumentParser(description="SEBT (Star Excursion Balance Test)")
    ap.add_argument("--front", default=None, help="front.csv yolu")
    ap.add_argument("--side", default=None, help="side.csv yolu")
    ap.add_argument("--height-m", type=float, default=None, dest="height_m")
    ap.add_argument("--directions", default=None)
    ap.add_argument("--legs", default=None)
    ap.add_argument("--limb-length-cm", type=float, default=None, dest="limb_length_cm")
    args = ap.parse_args()
    r = analyze(args.front, args.side, height_m=args.height_m,
                directions=args.directions, legs=args.legs,
                limb_length_cm=args.limb_length_cm)
    json.dump(r, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
