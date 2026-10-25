"""Document QA gates: blur, glare, skew, size — with explicit rejection reasons.

The same discipline as a data-quality pipeline, applied to images: an input
that fails a gate is rejected WITH a reason, never silently processed.
Thresholds were tuned on the synthetic golden set; they are intentionally
conservative (reject obvious junk, flag borderline cases for review).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

MIN_WIDTH = 480
MIN_HEIGHT = 300

# Tuned on data/golden: clean cards score ~55, GaussianBlur(3) cards ~15.
# Reject below the midpoint; borderline cases (15-40) deserve review, not silence.
SHARPNESS_REJECT = 25.0
# Fraction of near-white pixels: clean cards < 1%, glare-perturbed > 8%.
GLARE_REJECT = 0.05
# Degrees of rotation tolerated before rejection.
SKEW_REJECT = 4.0


@dataclass
class QaCheck:
    name: str
    value: float
    threshold: float
    passed: bool
    note: str = ""


@dataclass
class QaReport:
    ok: bool = True
    checks: list[QaCheck] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def _record(self, name: str, value: float, threshold: float, ok: bool, note: str = "") -> None:
        self.checks.append(QaCheck(name, round(float(value), 3), threshold, ok, note))
        if not ok:
            self.ok = False
            self.reasons.append(
                f"{name} failed: value={value:.3f} threshold={threshold} {note}".strip()
            )


def sharpness_score(img: Image.Image) -> float:
    """Std of edge intensity (Laplacian-style). Lower = blurrier."""
    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    return float(np.std(np.asarray(edges, dtype=np.float32)))


def glare_fraction(img: Image.Image) -> float:
    """Fraction of near-white pixels (specular highlights / overexposure)."""
    arr = np.asarray(img.convert("L"), dtype=np.uint8)
    return float((arr > 250).mean())


def estimate_skew(img: Image.Image, *, max_angle: float = 10.0, step: float = 0.5) -> float:
    """Estimate rotation via projection-profile variance.

    Text rows create strong horizontal structure: at the correct angle the
    row-sum variance of the edge map peaks. Search -max..+max degrees.
    """
    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    base = edges.rotate(0)
    best_angle, best_var = 0.0, -1.0
    angles = np.arange(-max_angle, max_angle + step / 2, step)
    for a in angles:
        rotated = edges.rotate(float(a), resample=Image.BILINEAR, fillcolor=0)
        var = float(np.var(np.asarray(rotated, dtype=np.float32).sum(axis=1)))
        if var > best_var:
            best_var, best_angle = var, float(a)
    del base
    return best_angle


def check_document(path: str | Path) -> QaReport:
    """Run all QA gates on one image file."""
    img = Image.open(path)
    report = QaReport()

    w, h = img.size
    report._record("width", w, MIN_WIDTH, w >= MIN_WIDTH, "px")
    report._record("height", h, MIN_HEIGHT, h >= MIN_HEIGHT, "px")

    sharp = sharpness_score(img)
    report._record("sharpness", sharp, SHARPNESS_REJECT, sharp >= SHARPNESS_REJECT, "std of edges")

    glare = glare_fraction(img)
    report._record("glare", glare, GLARE_REJECT, glare <= GLARE_REJECT, "near-white fraction")

    skew = estimate_skew(img)
    report._record("skew", skew, SKEW_REJECT, abs(skew) <= SKEW_REJECT, "degrees (abs)")

    return report
