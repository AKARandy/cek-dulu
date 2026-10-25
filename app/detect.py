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
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return {
            "text": text,
            "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
            "conf": round(float(conf), 3),
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


def assign_fields(lines: list[dict]) -> dict[str, list[int]]:
    """Map OCR lines to field keys via content rules + fixed-layout priors."""
    boxes: dict[str, list[int]] = {}

    # NIK: a 16-digit line high on the card.
    for ln in lines:
        if ln["bbox"][1] < NIK_MAX_Y and NIK_RE.search(ln["text"].replace(" ", "")):
            boxes["nik"] = ln["bbox"]
            break

    # Gender: the line that IS a gender word.
    for ln in lines:
        up = ln["text"].upper()
        if any(g in up for g in GENDER_WORDS):
            boxes["jenis_kelamin"] = ln["bbox"]
            break

    # Remaining left-column rows: nearest y-line, one line per row.
    used: set[int] = set()
    row_choice: dict[str, tuple[int, dict]] = {}
    for row_key, expected_y in LEFT_ROWS.items():
        if row_key == "jenis_kelamin" and "jenis_kelamin" in boxes:
            continue
        best_i, best_d = None, ROW_TOLERANCE
        for i, ln in enumerate(lines):
            if i in used:
                continue
            cy = (ln["bbox"][1] + ln["bbox"][3]) / 2
            d = abs(cy - expected_y)
            if d < best_d and ln["bbox"][0] < RIGHT_MIN_X:
                best_i, best_d = i, d
        if best_i is not None:
            used.add(best_i)
            row_choice[row_key] = (best_i, lines[best_i])

    # Label/value pairing: Paddle often reads "NAMA:" and the value as two
    # lines on the same row. If the chosen line is just the label, adopt the
    # nearest right-neighbor line on that row as the field's text.
    for row_key, (i, ln) in row_choice.items():
        chosen = ln
        text_norm = ln["text"].upper().replace(":", "").strip()
        label = ROW_LABELS.get(row_key)
        if label and (text_norm == label or text_norm.endswith(label)):
            cy = (ln["bbox"][1] + ln["bbox"][3]) / 2
            partner, partner_dx = None, 1e9
            for j, other in enumerate(lines):
                if j == i or j in used:
                    continue
                ocy = (other["bbox"][1] + other["bbox"][3]) / 2
                if (
                    abs(ocy - cy) <= 9
                    and other["bbox"][0] > ln["bbox"][2] - 5
                    and other["bbox"][2] < RIGHT_MIN_X
                ):
                    dx = other["bbox"][0] - ln["bbox"][2]
                    if 0 <= dx < partner_dx:
                        partner, partner_dx = other, dx
            if partner is not None:
                used.add(lines.index(partner))
                chosen = partner
        boxes[row_key] = chosen["bbox"]

    # Gol. darah: a tiny value riding to the right of the Jenis Kelamin row.
    if "jenis_kelamin" in boxes and any(k in boxes for k in ("nama",)):
        jk_box = boxes["jenis_kelamin"]
        cy = (jk_box[1] + jk_box[3]) / 2
        partner, partner_dx = None, 1e9
        for j, other in enumerate(lines):
            if other["bbox"] is jk_box or j in used:
                continue
            ocy = (other["bbox"][1] + other["bbox"][3]) / 2
            if (
                abs(ocy - cy) <= 9
                and other["bbox"][0] > jk_box[2] - 5
                and other["bbox"][2] < RIGHT_MIN_X
            ):
                dx = other["bbox"][0] - jk_box[2]
                if 0 <= dx < partner_dx and len(other["text"]) <= 4:
                    partner, partner_dx = other, dx
        if partner is not None:
            boxes["gol_darah"] = partner["bbox"]

    # Values sometimes ride in the same line as labels ("Nama: BUDI"); the
    # bbox still localizes the row, which is what downstream cropping needs.
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
