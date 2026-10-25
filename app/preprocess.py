"""Real-photo preprocessing: find the card, flatten it, fix the exposure.

Order matters and is deliberate:
1. Photometric fix FIRST - better contrast improves every later stage.
2. Fine-rotate the RAW capture (expand=True: the raw frame has margins, so
   nothing is cropped and no fill wedges appear).
3. THEN detect the card quad and perspective-warp it onto the canonical
   canvas. The warp re-rectifies residual tilt, so rotations never stack.

Rotating AFTER the warp is what produced rotated canvases with gray wedges;
that order is gone on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

CANON_W, CANON_H = 850, 540


@dataclass
class PreprocessReport:
    # "quad_warp" | "aspect_resize" | "aspect_resize_cropped" | "passthrough" | "passthrough_dim"
    method: str
    rotated_by: float = 0.0  # pre-warp fine rotation (degrees)
    photometric: bool = False

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "rotated_by": round(self.rotated_by, 2),
            "photometric": self.photometric,
        }


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order quad corners as [top-left, top-right, bottom-right, bottom-left]."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).squeeze()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _quad_area(quad: np.ndarray) -> float:
    x = quad[:, 0]
    y = quad[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _quad_is_cardlike(quad: np.ndarray) -> bool:
    """Reject contours that are not roughly a rectangle (angle sanity)."""
    c = quad.reshape(-1, 2).astype(np.float64)
    sides = [c[(i + 1) % 4] - c[i] for i in range(4)]
    angles = []
    for i in range(4):
        a, b = sides[i], sides[(i + 1) % 4]
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
        angles.append(abs(np.degrees(np.arccos(max(-1.0, min(1.0, cos))))))
    # interior angles of a rectangle are ~90 deg; allow generous warp
    return all(60 <= a <= 120 for a in angles)


def _find_card_quad(img: Image.Image) -> np.ndarray | None:
    """Largest plausible card-shaped 4-corner contour, or None.

    Tries plain edges first, then a CLAHE-boosted pass: dim captures with
    dark backgrounds yield no usable Canny edges until the contrast is
    lifted.
    """
    import cv2

    arr = np.asarray(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    frame_area = gray.shape[0] * gray.shape[1]

    def _quad_from(gray_img: np.ndarray) -> np.ndarray | None:
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 160)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_area = None, 0.0
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            area = cv2.contourArea(c)
            if area < 0.25 * frame_area:
                break
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            quad = _order_corners(approx.reshape(4, 2).astype(np.float32))
            if area > best_area and _quad_is_cardlike(quad):
                best_area, best = area, quad
        return best

    quad = _quad_from(gray)
    if quad is not None:
        return quad
    # Dim capture: boost local contrast and try again.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return _quad_from(clahe.apply(gray))


def _needs_light_fix(img: Image.Image) -> bool:
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    return bool(arr.mean() < 125 or arr.std() < 55)


def _apply_light_fix(img: Image.Image) -> Image.Image:
    import cv2

    arr = np.asarray(img.convert("RGB"))
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    lum, chrom_a, chrom_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lum = clahe.apply(lum)
    out = cv2.cvtColor(cv2.merge((lum, chrom_a, chrom_b)), cv2.COLOR_LAB2RGB)
    return Image.fromarray(out)


def _warp_to_canonical(img: Image.Image, quad: np.ndarray) -> Image.Image:
    import cv2

    dst = np.array(
        [[0, 0], [CANON_W - 1, 0], [CANON_W - 1, CANON_H - 1], [0, CANON_H - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(quad, dst)
    return Image.fromarray(
        cv2.warpPerspective(np.asarray(img), m, (CANON_W, CANON_H), flags=cv2.INTER_CUBIC)
    )


def _fine_rotate(img: Image.Image) -> tuple[Image.Image, float]:
    """Kill tilt on the RAW capture, expanding the canvas so nothing is lost.

    Only meaningful before the warp: the warp re-rectifies whatever tilt
    remains, and in-place rotation after the warp just shoves gray wedges
    into the corners.
    """
    from app.qa import estimate_skew

    total = 0.0
    for _ in range(2):
        skew = estimate_skew(img)
        if 0.8 <= abs(skew) <= 8.0:
            # estimate_skew returns the CORRECTION angle: rotate BY it.
            img = img.rotate(skew, resample=Image.BICUBIC, expand=True, fillcolor=(230, 230, 230))
            total += skew
        else:
            break
    return img, total


def preprocess(image_path: str | Path, out_path: str | Path) -> tuple[Path, PreprocessReport]:
    """Normalize one capture onto the canonical canvas. Returns (path, report)."""
    image_path = Path(image_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path).convert("RGB")

    # 1) contrast first: better edges for quad detection and OCR alike
    photometric = _needs_light_fix(img)
    if photometric:
        img = _apply_light_fix(img)

    # 2) fine-rotate the RAW frame (expand=True, nothing cropped)
    img, rotation = _fine_rotate(img)

    # 3) card quad -> warp; the warp is the only thing that touches geometry.
    #    Two-pass straightening: the detected quad's top/bottom edge angle IS
    #    the card's physical rotation (robust to background texture, unlike
    #    projection profiles). Pre-rotate by it, re-detect the quad on the
    #    straightened frame (edges are now axis-aligned and cleaner), then
    #    warp. Skipping this is what left cards tilted inside the canvas.
    quad = _find_card_quad(img)
    if quad is not None:
        import math

        top = quad[1] - quad[0]  # TL -> TR
        bot = quad[2] - quad[3]  # BL -> BR, same reading direction
        angle = math.degrees(math.atan2(top[1], top[0]) + math.atan2(bot[1], bot[0])) / 2
        if abs(angle) > 1.0:
            rotated = img.rotate(
                angle, resample=Image.BICUBIC, expand=True, fillcolor=(230, 230, 230)
            )
            re_quad = _find_card_quad(rotated)
            if re_quad is not None:
                img, quad = rotated, re_quad
        img = _warp_to_canonical(img, quad)
        method = "quad_warp"
    else:
        aspect = img.width / img.height
        if 1.25 <= aspect <= 1.90:
            # Card-shaped frame. It may BE the card edge-to-edge (render/scan)
            # or a photo with background bands; after resizing, try to crop the
            # card quad on the resized copy to shed any background.
            resized = img.resize((CANON_W, CANON_H), Image.LANCZOS)
            inner = _find_card_quad(resized)
            if inner is not None and _quad_area(inner) >= 0.55 * CANON_W * CANON_H:
                img = _warp_to_canonical(resized, inner)
                method = "aspect_resize_cropped"
            else:
                img = resized
                method = "aspect_resize"
        else:
            method = "passthrough"

    if method == "passthrough":
        report = PreprocessReport(
            method="passthrough" if not photometric else "passthrough_dim",
            rotated_by=rotation,
            photometric=photometric,
        )
    else:
        report = PreprocessReport(method=method, rotated_by=rotation, photometric=photometric)
    img.save(out_path)
    return out_path, report
