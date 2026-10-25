"""End-to-end pipeline: QA -> detect -> crop+OCR -> parse -> validate.

`gold=True` swaps model detection for the generator's ground-truth boxes and
stubs OCR with the generator's ground-truth text — the weight-free path used
by CI. Production path (`gold=False`) is identical except boxes/text come
from the models.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.ocr import StubOcr
from app.parse_fields import clean_nik, normalize, normalize_gender, parse_ttl_date, parse_ttl_place
from app.qa import check_document

SPEC_KEY_TO_TEXT: dict[str, Any] = {}  # populated by _spec_text()


def _spec_text(spec: Any) -> dict[str, str]:
    return {
        "nik": spec.nik,
        "nama": spec.nama,
        "ttl": f"{spec.ttl_tempat}, {spec.ttl_tanggal:%d-%m-%Y}",
        "jenis_kelamin": spec.jenis_kelamin,
        "gol_darah": spec.gol_darah,
        "alamat": spec.alamat,
        "rt_rw": spec.rt_rw,
        "kel_desa": spec.kel_desa,
        "kecamatan": spec.kecamatan,
        "agama": spec.agama,
        "status_perkawinan": spec.status_perkawinan,
        "pekerjaan": spec.pekerjaan,
        "kewarganegaraan": spec.kewarganegaraan,
        "berlaku_hingga": spec.berlaku_hingga,
    }


def line_skew_deg(lns: list[dict]) -> float | None:
    """Slope of OCR text-row centers in degrees; None when too few lines."""
    import math

    left = [ln for ln in lns if ln["bbox"][0] < 550 and ln["bbox"][2] < 600]
    if len(left) < 6:
        return None
    xs = [(ln["bbox"][0] + ln["bbox"][2]) / 2 for ln in left]
    ys = [(ln["bbox"][1] + ln["bbox"][3]) / 2 for ln in left]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False)) / denom
    return math.degrees(math.atan(slope))


def run(
    image_path: str | Path,
    *,
    gold_entry: dict | None = None,
    gold_spec: Any = None,
    workdir: str | Path | None = None,
    skip_preprocess: bool = False,
) -> dict:
    """Process one KTP image. Returns the full result dict (JSON-safe)."""
    image_path = Path(image_path)
    workdir = Path(workdir) if workdir else image_path.parent / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {"image": image_path.name}

    # Real captures: flatten the card onto the canonical canvas first.
    # Gold mode skips this: synthetic cards are already canonical.
    if gold_entry is None and not skip_preprocess:
        from app.preprocess import preprocess

        normalized, prep_report = preprocess(image_path, workdir / "normalized.png")
        result["preprocess"] = prep_report.as_dict()
        image_path = normalized

    qa = check_document(image_path)
    result["qa"] = asdict(qa)

    def _reject() -> dict:
        result["extracted"] = None
        result["validation"] = None
        result["rejected"] = True
        return result

    # Hard QA failures (anything but skew) reject immediately. A skew-only
    # failure is provisional: on the canonical canvas the projection profile
    # reads the true residual tilt, so the model path rotates by exactly that
    # amount, re-runs QA, and adopts the flatter image.
    if not qa.ok and any(c.name != "skew" for c in qa.checks if not c.passed):
        return _reject()
    if not qa.ok and gold_entry is not None:
        return _reject()  # gold cards are canonical renders: skew fail is real
    if not qa.ok:
        _skew_val = next(c.value for c in qa.checks if c.name == "skew" and not c.passed)
        from PIL import Image as _Img

        _img = _Img.open(image_path).convert("RGB")
        # estimate_skew returns the CORRECTION angle: rotate BY it.
        _rotated = _img.rotate(
            _skew_val, resample=_Img.BICUBIC, expand=False, fillcolor=(228, 228, 228)
        )
        _retry_path = workdir / "deskewed.png"
        _rotated.save(_retry_path)
        _qa2 = check_document(_retry_path)
        if _qa2.ok or (
            _qa2.ok is False
            and next((c.value for c in _qa2.checks if c.name == "skew"), 99) < abs(_skew_val)
        ):
            image_path, qa = _retry_path, _qa2
            result["preprocess"]["deskew_retry"] = round(_skew_val, 2)
            result["qa"] = asdict(qa)
    if not qa.ok:
        return _reject()

    # --- boxes + text: gold (generator truth) or model path ---
    if gold_entry is not None:
        boxes = {k: list(v) for k, v in (gold_entry.get("bboxes") or {}).items()}
        if gold_spec is not None:
            stub = StubOcr(_spec_text(gold_spec))
            fields = {
                k: {"text": stub.read(workdir / f"{k}.png", k)[0], "conf": 1.0, "bbox": v}
                for k, v in boxes.items()
            }
    else:
        from app.detect import assign_fields, ground_photo_box, ocr_lines, photo_box_from_lines

        lines = ocr_lines(image_path)

        def refine_nik(lns: list[dict]) -> list[dict]:
            """Arbitrate the NIK reading.

            Candidates: the full-image read plus per-position-majority votes
            over multiple enhancement variants (scale x3/x5, grayscale
            autocontrast, sharpening) of an upscaled NIK-row crop. The winner
            is the candidate whose decoded birth date matches a date printed
            on the card (the NIK encodes the TTL date by law). With no date
            evidence the full 16-digit read stands, and any contradiction
            surfaces later as a validation issue.
            """
            import re as _re
            from collections import Counter

            from PIL import Image, ImageFilter, ImageOps

            from app.parse_fields import parse_ttl_date
            from validators.nik import validate_nik

            cand = [
                ln
                for ln in lns
                if _re.search(r"\d{12,}", ln["text"].replace(" ", "")) and ln["bbox"][1] < 260
            ]
            if not cand:
                return lns
            target = min(cand, key=lambda ln: ln["bbox"][1])

            def digits_of(s: str) -> str:
                return _re.sub(r"\D", "", s)

            full_digits = digits_of(target["text"])[:16]
            card_dates = [d for ln in lns if (d := parse_ttl_date(ln["text"]))]

            def score(nik16: str) -> int:
                check = validate_nik(nik16)
                if not check.ok or check.birth_date is None:
                    return 0
                if card_dates and check.birth_date in card_dates:
                    return 2  # decodes to a date printed on the card
                if not card_dates:
                    return 1
                return 0  # contradicts every printed date

            candidates = [full_digits] if len(full_digits) >= 16 else []

            box = target["bbox"]
            img = Image.open(image_path).convert("RGB")
            pad = 8
            crop = img.crop(
                (
                    max(0, box[0] - pad),
                    max(0, box[1] - pad),
                    min(img.width, box[2] + pad),
                    min(img.height, box[3] + pad),
                )
            )
            variant_reads: list[str] = []
            vi = 0
            for scale in (3, 5):
                big = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
                for im in (
                    big,
                    ImageOps.autocontrast(big.convert("L")).convert("RGB"),
                    big.filter(ImageFilter.SHARPEN),
                ):
                    vp = workdir / f"nik_strip_{vi}.png"
                    vi += 1
                    im.save(vp)
                    for sl in ocr_lines(vp):
                        d16 = digits_of(sl["text"])[:16]
                        if len(d16) == 16 and d16 not in variant_reads:
                            variant_reads.append(d16)
            if len(variant_reads) >= 3:
                voted = "".join(
                    Counter(pos).most_common(1)[0][0] for pos in zip(*variant_reads, strict=False)
                )
                if len(voted) == 16 and validate_nik(voted).ok:
                    candidates.append(voted)
            for d16 in variant_reads:
                if d16 not in candidates:
                    candidates.append(d16)

            if not candidates:
                return lns
            best = max(candidates, key=score)
            if score(best) == 0 and len(full_digits) >= 16:
                best = full_digits  # no evidence against it: keep the full read
            if len(best) == 16 and best != digits_of(target["text"]):
                target["text"] = best
                target["conf"] = max(target["conf"], 0.95)
            return lns

        lines = refine_nik(lines)
        by_bbox = {tuple(ln["bbox"]): ln for ln in lines}

        boxes = assign_fields(lines)
        photo, photo_source = ground_photo_box(image_path)
        if photo_source == "layout_prior":
            adaptive = photo_box_from_lines(lines)
            if adaptive:
                photo, photo_source = adaptive, "text_density"
        boxes["foto"] = photo
        fields = {}
        for key, bbox in boxes.items():
            ln = by_bbox.get(tuple(bbox))
            if ln is not None:
                fields[key] = {"text": ln["text"], "conf": ln["conf"], "bbox": bbox}
            else:
                fields[key] = {"text": "", "conf": 0.0, "bbox": bbox}

    # --- parse + validate ---
    from validators.nik import cross_check_fields, validate_nik

    def strip_label(text: str, key: str) -> str:
        """Drop a leading 'LABEL:' if Paddle merged label and value in one line."""
        label = {
            "ttl": "TEMPAT/TGL LAHIR",
            "nama": "NAMA",
            "alamat": "ALAMAT",
            "rt_rw": "RT/RW",
            "kel_desa": "KEL/DESA",
            "kecamatan": "KECAMATAN",
            "agama": "AGAMA",
            "status_perkawinan": "STATUS PERKAWINAN",
            "pekerjaan": "PEKERJAAN",
            "kewarganegaraan": "KEWARGANEGARAAN",
            "berlaku_hingga": "BERLAKU HINGGA",
        }.get(key)
        if not label:
            return text
        up = text.upper()
        for sep in (
            ":",
            " ",
        ):
            prefix = label + sep
            if up.startswith(prefix):
                return text[len(prefix) :].strip(" :")
        return text

    for key, v in fields.items():
        v["text"] = strip_label(v.get("text", ""), key).lstrip(" :")

    # TTL display: rebuild as 'PLACE, DD-MM-YYYY' with OCR digit fixes in the
    # place part (LE8AK -> LEBAK); the date itself parses as-is.
    _ttl = fields.get("ttl", {}).get("text", "")
    _place = parse_ttl_place(_ttl)
    _d = parse_ttl_date(_ttl)
    if _place and _d:
        fields["ttl"]["text"] = f"{_place}, {_d:%d-%m-%Y}"

    nik_raw = fields.get("nik", {}).get("text", "")
    nik = clean_nik(nik_raw)
    ttl_text = fields.get("ttl", {}).get("text", "")
    ttl_date = parse_ttl_date(ttl_text)
    jk = normalize_gender(fields.get("jenis_kelamin", {}).get("text", ""))

    check = validate_nik(nik)
    if ttl_date or jk:
        check = cross_check_fields(check, ttl_date=ttl_date, jenis_kelamin=jk)

    result["rejected"] = False
    result["extracted"] = {
        key: {"text": normalize(v["text"]), "conf": v["conf"], "bbox": v["bbox"]}
        for key, v in fields.items()
    }
    # A card that passed QA but yielded no readable NIK must still produce a
    # structured result, never a KeyError downstream.
    result["extracted"].setdefault("nik", {"text": nik, "conf": 0.0, "bbox": None})
    result["extracted"]["nik"]["text"] = nik  # cleaned digits, not re-normalized
    result["validation"] = {
        "ok": check.ok,
        "province": check.province,
        "birth_date": check.birth_date.isoformat() if check.birth_date else None,
        "gender": check.gender,
        "day_unknown": check.day_unknown,
        "issues": check.issues,
    }
    # --- Authenticity notes (signals, not verdicts) ---
    notes: list[str] = []
    berlaku = (result["extracted"].get("berlaku_hingga") or {}).get("text", "")
    has_date = bool(re.search(r"\d", berlaku))
    if berlaku and has_date and "SEUMUR" not in berlaku.upper():
        notes.append(
            f"non-lifetime validity ({berlaku!r}): e-KTP is SEUMUR HIDUP; "
            "expired-date validity suggests an old/blank/non-standard card"
        )
    if check.province:
        header_province = None
        for v in fields.values():
            up = (v.get("text") or "").upper()
            if "PROVINSI" in up:
                header_province = up.replace("PROVINSI", "").strip()
                break
        if (
            header_province
            and header_province not in check.province
            and check.province not in header_province
        ):
            notes.append(
                f"header province {header_province!r} disagrees with "
                f"NIK province {check.province!r}"
            )
    result["authenticity"] = notes

    result["boxes"] = boxes
    # Boxes live in the coordinate space of the image actually OCR'd
    # (preprocessed + deskewed); overlays must render on THIS image.
    result["processed_image"] = str(image_path)
    # Which backend produced the photo box on the model path; None on gold mode.
    result["photo_box_source"] = locals().get("photo_source")
    return result
