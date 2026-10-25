"""Field localization for the KTP.

Two complementary signals, each doing what it is good at:
- Florence-2 phrase grounding finds the PHOTO box (IoU ~0.98 on the golden
  set) — a visual object, which is what grounding is for.
- PaddleOCR text-line detection finds every text line WITH its box; fixed
  document layout + content rules then assign lines to fields. Dense text
  regions are exactly what OCR detectors are built for.

This split mirrors production eKYC stacks: text detector + document parser,
with a detector only where a non-text object matters.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

try:  # Windows DLL order: torch must load before paddle/cv2, or shm.dll fails.
    import torch  # noqa: F401
except ImportError:  # allows validator/tests to import this module torch-free
    torch = None  # type: ignore[assignment]

_MODEL: Any = None
_PROCESSOR: Any = None
_DEVICE = "cpu"

LEFT_ROWS: dict[str, float] = {
    "nama": 107,
    "ttl": 131,
    "jenis_kelamin": 155,
    "alamat": 179,
    "rt_rw": 203,
    "kel_desa": 227,
    "kecamatan": 251,
    "agama": 293,
    "status_perkawinan": 317,
    "pekerjaan": 341,
    "kewarganegaraan": 365,
    "berlaku_hingga": 389,
}
NIK_MAX_Y = 95
RIGHT_MIN_X = 550
ROW_TOLERANCE = 24

NIK_RE = re.compile(r"\d{4}[\s.-]?\d{4}[\s.-]?\d{4}[\s.-]?\d{4}")
DATE_RE = re.compile(r"\d{1,2}[-/. ]\d{1,2}[-/. ]\d{4}")
GENDER_WORDS = ("LAKI-LAKI", "PEREMPUAN")

ROW_LABELS: dict[str, str] = {
    "nama": "NAMA",
    "ttl": "TEMPAT/TGL LAHIR",
    "alamat": "ALAMAT",
    "rt_rw": "RT/RW",
    "kel_desa": "KEL/DESA",
    "kecamatan": "KECAMATAN",
    "agama": "AGAMA",
    "status_perkawinan": "STATUS PERKAWINAN",
    "pekerjaan": "PEKERJAAN",
    "kewarganegaraan": "KEWARGANEGARAAN",
    "berlaku_hingga": "BERLAKU HINGGA",
}


def _model_dir() -> Path:
    d = Path(os.environ.get("CEKDULU_MODEL_DIR", ".models"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_model(model_id: str = "microsoft/Florence-2-base") -> None:
    """Load Florence-2 once (photo grounding). Offline after first download."""
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    _PROCESSOR = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=_model_dir()
    )
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=_model_dir(), torch_dtype=torch.float32
    ).to(_DEVICE)
    _MODEL.eval()


def ground_photo_box(image_path: str | Path) -> tuple[list[int], str]:
    """Photo box with provenance: (box, "florence" | "layout_prior").

    Grounding is an enhancement; the card layout guarantees the position, so
    any grounding failure falls back to the layout prior, and the caller
    records which one actually produced the box.
    """
    try:
        grounded = _florence_photo(image_path)
        if grounded is not None:
            return grounded, "florence"
    except Exception:
        pass
    return [610, 100, 810, 350], "layout_prior"


def _florence_photo(image_path: str | Path) -> list[int] | None:
    if torch is None:
        return None
    from PIL import Image

    load_model()
    image = Image.open(image_path).convert("RGB")

    inputs = _PROCESSOR(
        images=image,
        text="<CAPTION_TO_PHRASE_GROUNDING>photo</CAPTION_TO_PHRASE_GROUNDING>",
        return_tensors="pt",
    ).to(_DEVICE)
    with torch.no_grad():
        generated = _MODEL.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
    decoded = _PROCESSOR.batch_decode(generated, skip_special_tokens=False)[0]
    result = _PROCESSOR.post_process_generation(
        decoded, task="<CAPTION_TO_PHRASE_GROUNDING>", image_size=(image.width, image.height)
    )
    boxes = [
        b
        for b in result.get("<CAPTION_TO_PHRASE_GROUNDING>", {}).get("bboxes", [])
        if b[0] >= RIGHT_MIN_X
    ]
    if not boxes:
        return None
    biggest = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return [int(v) for v in biggest]


_PADDLE_LINES: Any = None


def ocr_lines(image_path: str | Path) -> list[dict]:
    """PaddleOCR text lines: [{text, bbox[x0,y0,x1,y1], conf}] top-to-bottom.

    Supports both paddleocr 3.x (predict/rec_texts) and 2.x (ocr/cls) APIs.
    """
    global _PADDLE_LINES
    if _PADDLE_LINES is None:
        from paddleocr import PaddleOCR  # type: ignore

        try:
            _PADDLE_LINES = PaddleOCR(lang="en")  # 3.x signature
        except TypeError:
            _PADDLE_LINES = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)  # 2.x

    def _from_poly(text: str, conf: float, poly: Any) -> dict:
        import math as _math

        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        # The quad's top edge gives the line's own tilt: the physical card
        # skew, measured per line, robust to two-column layout offsets.
        angle = 0.0
        if len(poly) >= 2:
            dx = poly[1][0] - poly[0][0]
            dy = poly[1][1] - poly[0][1]
            if abs(dx) > 1e-3:
                angle = _math.degrees(_math.atan2(dy, dx))
        return {
            "text": text,
            "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
            "conf": round(float(conf), 3),
            "angle": round(angle, 2),
        }

    lines: list[dict] = []
    try:  # paddleocr 3.x
        for res in _PADDLE_LINES.predict(str(image_path)):
            zipped = zip(res["rec_texts"], res["rec_scores"], res["dt_polys"], strict=False)
            for text, score, poly in zipped:
                lines.append(_from_poly(text, score, poly))
    except (AttributeError, KeyError, NotImplementedError):  # paddleocr 2.x fallback
        result = _PADDLE_LINES.ocr(str(image_path), cls=True)
        if result and result[0]:
            for item in result[0]:
                lines.append(_from_poly(item[1][0], item[1][1], item[0]))
    lines.sort(key=lambda ln: ln["bbox"][1])
    return lines


def assign_fields(lines: list[dict], *, shear: float = 0.0) -> dict[str, list[int]]:
    """Map OCR lines to field keys.

    Content first, geometry second: real cards and synthetic renders sit at
    different absolute positions, but every Indonesian KTP prints the same
    labels. Values live in their own column; labels pair to values by a
    greedy one-to-one nearest-row match, which survives baseline offsets and
    dense row spacing where a fixed tolerance collides across rows.
    """
    boxes: dict[str, list[int]] = {}
    used: set[int] = set()

    def norm_text(s: str) -> str:
        return re.sub(r"[^A-Z0-9/]", "", s.upper())

    def center_y(bbox: list[int]) -> float:
        return (bbox[1] + bbox[3]) / 2

    def level_y(ln: dict) -> float:
        """Row center in shear-leveled coordinates.

        On a tilted card the value column (larger x) sits systematically
        lower than the label column; subtracting shear*x_mid flattens the
        rows so label-value gaps become small and uniform.
        """
        return center_y(ln["bbox"]) - shear * (ln["bbox"][0] + ln["bbox"][2]) / 2

    header_prefixes = ("PROVINSI", "KABUPATEN", "KOTA", "REPUBLIK", "INDONESIA")
    label_to_key = {re.sub(r"[^A-Z0-9/]", "", v.upper()): k for k, v in ROW_LABELS.items()}
    # Prefix anchors survive OCR mangling ("Tempatagahir", "Alamal", "RURW").
    # Longest first so KECAMATAN wins over KEL/DESA-style prefixes.
    anchor_to_key = sorted(
        {
            "NAMA": "nama",
            "TEMPA": "ttl",
            "JENIS": "jenis_kelamin",
            "ALAMA": "alamat",
            "RT/RW": "rt_rw",
            "RTRW": "rt_rw",
            "KEL/D": "kel_desa",
            "KELDE": "kel_desa",
            "KECAM": "kecamatan",
            "AGAMA": "agama",
            "STATU": "status_perkawinan",
            "PEKER": "pekerjaan",
            "KEWAR": "kewarganegaraan",
            "BERLA": "berlaku_hingga",
            "GOL": "gol_darah_label",
        }.items(),
        key=lambda kv: -len(kv[0]),
    )

    def classify(ln: dict) -> str:
        n = norm_text(ln["text"])
        if not n:
            return "empty"
        if any(n.startswith(h) for h in header_prefixes):
            return "header"
        for label in label_to_key:
            if n == label or n.startswith(label):
                return "label"
        for anchor, _key in anchor_to_key:
            if len(n) >= 4 and (n.startswith(anchor) or anchor.startswith(n) and len(n) >= 5):
                return "label"
        return "value"

    # --- NIK: 16 digits anywhere; prefer the upper half ---
    nik_cands = [ln for ln in lines if NIK_RE.search(ln["text"].replace(" ", ""))]
    if nik_cands:
        mid = sum(ln["bbox"][3] for ln in lines) / len(lines)
        upper = [ln for ln in nik_cands if ln["bbox"][1] < mid] or nik_cands
        upper.sort(key=lambda ln: ln["bbox"][1])
        chosen = upper[0]
        boxes["nik"] = chosen["bbox"]
        used.add(lines.index(chosen))

    # --- Classify every line ---
    roles = [classify(ln) for ln in lines]

    def anchor_key(text: str) -> str | None:
        n = norm_text(text)
        for label, key in sorted(label_to_key.items(), key=lambda L: -len(L)):
            if n == label or n.startswith(label):
                return key
        for anchor, key in anchor_to_key:
            if len(n) >= 4 and (n.startswith(anchor) or (anchor.startswith(n) and len(n) >= 5)):
                return None if key == "gol_darah_label" else key
        return None

    label_line_of_key: dict[str, int] = {}
    for i, (ln, role) in enumerate(zip(lines, roles, strict=False)):
        if role != "label" or i in used:
            continue
        key = anchor_key(ln["text"])
        if key and key not in boxes and key not in label_line_of_key:
            label_line_of_key[key] = i
            boxes[key] = ln["bbox"]  # provisional: the label row itself

    # --- Greedy one-to-one pairing: labels to value-column lines by row ---
    value_idx = [
        i
        for i, (ln, role) in enumerate(zip(lines, roles, strict=False))
        if role == "value" and i not in used and 100 <= ln["bbox"][0] <= 430
    ]
    # Pair labels to values in shear-leveled coordinates: a value belongs to
    # the label whose leveled row is nearest ABOVE it (small positive gap =
    # the usual baseline offset). Greedy on smallest gap keeps it one-to-one.
    claimed: set[str] = set()
    assignments: list[tuple[float, str, int]] = []
    for vi in value_idx:
        vy = level_y(lines[vi])
        best_gap, best_key = None, None
        for key, li in label_line_of_key.items():
            if key in claimed:
                continue
            gap = vy - level_y(lines[li])
            if -3 <= gap <= 20 and (best_gap is None or gap < best_gap):
                best_gap, best_key = gap, key
        if best_key is not None:
            assignments.append((best_gap, best_key, vi))
    assignments.sort(key=lambda t: t[0])
    for _gap, key, vi in assignments:
        if key in claimed or vi in used:
            continue
        boxes[key] = lines[vi]["bbox"]
        used.add(vi)
        claimed.add(key)

    # --- Gender: a line that IS a gender word ---
    if "jenis_kelamin" not in boxes:
        for i, ln in enumerate(lines):
            if i not in used and any(g in ln["text"].upper() for g in GENDER_WORDS):
                boxes["jenis_kelamin"] = ln["bbox"]
                used.add(i)
                break

    # --- Fallback: synthetic-geometry priors for unclaimed fields ---
    for row_key, expected_y in LEFT_ROWS.items():
        if row_key in boxes:
            continue
        best_i, best_d = None, ROW_TOLERANCE
        for i, ln in enumerate(lines):
            if i in used or roles[i] != "value":
                continue
            d = abs(center_y(ln["bbox"]) - expected_y)
            if d < best_d and ln["bbox"][0] < RIGHT_MIN_X:
                best_i, best_d = i, d
        if best_i is not None:
            used.add(best_i)
            boxes[row_key] = lines[best_i]["bbox"]

    # --- Gol. darah: tiny value right of the Jenis Kelamin row ---
    if "jenis_kelamin" in boxes:
        jk_y = center_y(boxes["jenis_kelamin"])
        best, best_dx = None, 1e9
        for i, ln in enumerate(lines):
            if i in used:
                continue
            if abs(center_y(ln["bbox"]) - jk_y) <= 14 and len(ln["text"].strip()) <= 4:
                dx = ln["bbox"][0] - boxes["jenis_kelamin"][2]
                if 0 <= dx < best_dx:
                    best, best_dx = ln, dx
        if best is not None:
            boxes["gol_darah"] = best["bbox"]

    return boxes


def detect_fields(image_path: str | Path, *, include_photo: bool = True) -> dict[str, list[int]]:
    """Full field localization: OCR lines + rules, plus Florence-2 photo box."""
    boxes = assign_fields(ocr_lines(image_path))
    if include_photo:
        photo = ground_photo_box(image_path)
        if photo:
            boxes["foto"] = photo
    return boxes


def detect_fields_gold(ground_truth_entry: dict) -> dict[str, list[int]]:
    """Gold mode: generator's own boxes, for benchmarks and weight-free tests."""
    return {k: list(v) for k, v in (ground_truth_entry.get("bboxes") or {}).items()}


def iou(a: list[int], b: list[int]) -> float:
    """Intersection-over-union of two [x0,y0,x1,y1] boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0
