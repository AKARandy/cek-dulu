"""OCR post-processing: normalize text, extract dates, clean the NIK."""

from __future__ import annotations

import re
from datetime import date


def normalize(text: str) -> str:
    """Uppercase, collapse whitespace, drop stray OCR underscores."""
    text = text.replace("_", " ").strip()
    return re.sub(r"\s+", " ", text).upper()


def clean_nik(text: str) -> str:
    digits = re.sub(r"\D", "", text)
    # OCR often splits/confuses digits; keep exactly 16 if we have them.
    return digits[:16] if len(digits) >= 16 else digits


DATE_RE = re.compile(r"(\d{1,2})[-/. ](\d{1,2})[-/. ](\d{4})")


def parse_ttl_date(text: str) -> date | None:
    """Extract the DD-MM-YYYY birth date from a 'PLACE, DD-MM-YYYY' string."""
    m = DATE_RE.search(text)
    if not m:
        return None
    d_, m_, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return date(y, m_, d_)
    except ValueError:
        return None


# Indonesian place names are alphabetic; OCR swaps letters for lookalike
# digits (B->8, O->0, I->1, S->5, G->6). Only applied to the place segment.
_OCR_DIGIT_MAP = str.maketrans({"8": "B", "0": "O", "1": "I", "5": "S", "6": "G"})


def fix_place_glyphs(place: str) -> str:
    """Map OCR digit/letter confusions inside alphabetic place tokens."""
    fixed = []
    for tok in place.split():
        if any(c.isalpha() for c in tok) and any(c.isdigit() for c in tok):
            tok = tok.translate(_OCR_DIGIT_MAP)
        fixed.append(tok)
    return " ".join(fixed)


def parse_ttl_place(text: str) -> str | None:
    m = DATE_RE.search(text)
    if not m:
        return normalize(text) or None
    place = text[: m.start()].strip().rstrip(",.").strip()
    return fix_place_glyphs(normalize(place)) or None


GENDER_MAP = {
    "LAKI-LAKI": "LAKI-LAKI",
    "LAKILAKI": "LAKI-LAKI",
    "L": "LAKI-LAKI",
    "PRIA": "LAKI-LAKI",
    "PEREMPUAN": "PEREMPUAN",
    "P": "PEREMPUAN",
    "WANITA": "PEREMPUAN",
}


def normalize_gender(text: str) -> str | None:
    return GENDER_MAP.get(normalize(text).replace(" ", ""))


def normalized_distance(a: str, b: str) -> float:
    """Cheap normalized Levenshtein for field-accuracy benchmarking."""
    a, b = normalize(a), normalize(b)
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 1.0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[lb] / max(la, lb)
