"""Per-crop OCR with a pluggable engine: paddle (default) | tesseract | stub.

The stub engine exists so the API and tests run anywhere without model
downloads; it accepts pre-supplied values (gold mode) and returns them
verbatim with confidence 1.0.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENGINE: str | None = None
_PADDLE = None


class StubOcr:
    """Returns pre-supplied text per field — used by tests and gold mode."""

    def __init__(self, supplied: dict[str, str] | None = None):
        self.supplied = supplied or {}

    def read(self, crop: Path, field_key: str) -> tuple[str, float]:
        return self.supplied.get(field_key, ""), 1.0


def engine_name() -> str:
    return _ENGINE or os.environ.get("CEKDULU_OCR_ENGINE", "paddle")


def _load_paddle():
    global _PADDLE, _ENGINE
    if _PADDLE is not None:
        return _PADDLE
    from paddleocr import PaddleOCR  # type: ignore

    _PADDLE = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    _ENGINE = "paddle"
    return _PADDLE


def ocr_crop(
    crop_path: str | Path, field_key: str, *, engine: str | None = None
) -> tuple[str, float]:
    """OCR one crop; returns (text, confidence 0-1)."""
    engine = engine or engine_name()
    if engine == "stub":
        return "", 0.0  # StubOcr instance should be used instead for real values
    if engine == "tesseract":
        import pytesseract
        from PIL import Image

        img = Image.open(crop_path)
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config="--psm 7")
        words, confs = [], []
        for w, c in zip(data["text"], data["conf"], strict=False):
            if w.strip() and int(c) > 0:
                words.append(w)
                confs.append(int(c))
        text = " ".join(words)
        conf = sum(confs) / len(confs) / 100 if confs else 0.0
        return text, round(conf, 3)
    # default: paddle
    p = _load_paddle()
    result = p.ocr(str(crop_path), cls=True)
    if not result or not result[0]:
        return "", 0.0
    lines = []
    confs = []
    for line in result[0]:
        text, (conf,) = line[1][0], (line[1][1],)
        lines.append(text)
        confs.append(conf)
    return " ".join(lines), round(sum(confs) / len(confs), 3)


def crop_and_ocr(
    image_path: str | Path,
    boxes: dict[str, list[int]],
    out_dir: str | Path,
    *,
    engine: str | None = None,
) -> dict[str, dict]:
    """Crop each box from the image and OCR it. Returns {field: {text, conf, bbox}}."""
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path)
    results: dict[str, dict] = {}
    for key, (x0, y0, x1, y1) in boxes.items():
        pad = 3
        crop = img.crop(
            (
                max(0, x0 - pad),
                max(0, y0 - pad),
                min(img.width, x1 + pad),
                min(img.height, y1 + pad),
            )
        )
        crop_path = out_dir / f"{key}.png"
        crop.save(crop_path)
        text, conf = ocr_crop(crop_path, key, engine=engine)
        results[key] = {"text": text, "conf": conf, "bbox": [x0, y0, x1, y1]}
    return results
