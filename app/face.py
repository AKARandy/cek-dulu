"""Face similarity via an ArcFace-family ONNX model (lazy, offline).

License note for the README: InsightFace pretrained weights (buffalo_l /
w600k line) are provided for non-commercial research use. Code here is
original. The model downloads once from the official insightface release
and is cached in CEKDULU_MODEL_DIR.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Any

_SESSION: Any = None
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"


def model_path() -> Path:
    d = Path(os.environ.get("CEKDULU_MODEL_DIR", ".models"))
    d.mkdir(parents=True, exist_ok=True)
    return d / "w600k_r50.onnx"


def load_model() -> None:
    global _SESSION
    if _SESSION is not None:
        return
    import urllib.request

    import onnxruntime as ort

    path = model_path()
    if not path.exists():
        zip_path = path.parent / "buffalo_l.zip"
        if not zip_path.exists():
            urllib.request.urlretrieve(BUFFALO_URL, zip_path)  # noqa: S310 (pinned https source)
        with zipfile.ZipFile(zip_path) as zf:
            member = next(m for m in zf.namelist() if m.endswith("w600k_r50.onnx"))
            path.write_bytes(zf.read(member))
        zip_path.unlink(missing_ok=True)
    _SESSION = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _preprocess(crop_path: str | Path) -> Any:
    import numpy as np
    from PIL import Image

    img = Image.open(crop_path).convert("RGB").resize((112, 112))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return arr.transpose(2, 0, 1)[None]


def embed(crop_path: str | Path) -> Any:
    import numpy as np

    load_model()
    x = _preprocess(crop_path)
    input_name = _SESSION.get_inputs()[0].name
    (emb,) = _SESSION.run(None, {input_name: x})
    emb = emb[0]
    return emb / np.linalg.norm(emb)


def similarity(crop_a: str | Path, crop_b: str | Path) -> float:
    """Cosine similarity of two face crops, in [-1, 1]."""
    import numpy as np

    a, b = embed(crop_a), embed(crop_b)
    return float(np.dot(a, b))


def match(crop_a: str | Path, crop_b: str | Path, *, threshold: float | None = None) -> dict:
    """Decision-ready match result with the threshold stated, not hidden."""
    threshold = (
        threshold
        if threshold is not None
        else float(os.environ.get("CEKDULU_MATCH_THRESHOLD", "0.35"))
    )
    score = similarity(crop_a, crop_b)
    return {
        "similarity": round(score, 4),
        "threshold": threshold,
        "same_person": score >= threshold,
        "verdict": "MATCH" if score >= threshold else "DIFFERENT",
    }
