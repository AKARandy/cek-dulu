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
from app.parse_fields import clean_nik, normalize, normalize_gender, parse_ttl_date
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
    # failure is provisional: the projection-profile estimator is fooled by
    # guilloche texture, so the model path re-verifies tilt against the OCR
    # line boxes (the text rows themselves) before deciding.
    if not qa.ok and any(c.name != "skew" for c in qa.checks if not c.passed):
        return _reject()
    if not qa.ok and gold_entry is not None:
        return _reject()  # gold cards are canonical renders: skew fail is real

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
        from app.detect import assign_fields, ground_photo_box, ocr_lines

        lines = ocr_lines(image_path)

        # Text rows are the referee: if they still tilt after preprocessing,
        # rotate by the measured slope and re-OCR once. Bounded to one retry.
        ls0 = line_skew_deg(lines)
        if ls0 is not None and abs(ls0) > 1.2:
            from PIL import Image as _Img

            img = _Img.open(image_path).convert("RGB")
            img = img.rotate(ls0, resample=_Img.BICUBIC, expand=False, fillcolor=(228, 228, 228))
            corrected = workdir / "deskewed.png"
            img.save(corrected)
            re_lines = ocr_lines(corrected)
            if line_skew_deg(re_lines) is not None and abs(line_skew_deg(re_lines)) < abs(ls0):
                image_path = corrected
                lines = re_lines

        # NIK refine: small fonts on guilloche misread at full-image scale;
        # crop the NIK row, upscale, and re-OCR just that strip.
        def refine_nik(lns: list[dict]) -> list[dict]:
            import re as _re

            from PIL import Image as _Img

            cand = [
                ln
                for ln in lns
                if _re.search(r"\d{12,}", ln["text"].replace(" ", "")) and ln["bbox"][1] < 260
            ]
            if not cand:
                return lns
            box = min(cand, key=lambda ln: ln["bbox"][1])["bbox"]
            img = _Img.open(image_path).convert("RGB")
            pad = 6
            crop = img.crop(
                (
                    max(0, box[0] - pad),
                    max(0, box[1] - pad),
                    min(img.width, box[2] + pad),
                    min(img.height, box[3] + pad),
                )
            )
            crop = crop.resize((crop.width * 3, crop.height * 3), _Img.LANCZOS)
            strip = workdir / "nik_strip.png"
            crop.save(strip)
            strip_lines = ocr_lines(strip)
            best = ""
            for sl in strip_lines:
                digits = _re.sub(r"\D", "", sl["text"])
                if len(digits) > len(best):
                    best = digits
            if len(best) >= 16:
                nik16 = best[:16]
                for ln in lns:
                    if ln is cand[0]:
                        ln["text"] = nik16
                        ln["conf"] = max(ln["conf"], 0.95)
                        break
            return lns

        lines = refine_nik(lines)
        by_bbox = {tuple(ln["bbox"]): ln for ln in lines}

        # Text-based skew verdict: the projection-profile estimator can be
        # fooled by guilloche texture. The OCR line boxes ARE the text rows,
        # so their slope is the ground truth for tilt.
        if not result["qa"]["ok"] and all(
            c["name"] == "skew" for c in result["qa"]["checks"] if not c["passed"]
        ):
            ls = line_skew_deg(lines)
            if ls is not None and abs(ls) <= 4.0:
                for c in result["qa"]["checks"]:
                    if c["name"] == "skew":
                        c["passed"] = True
                        c["note"] = f"texture estimate overruled by OCR-line slope {ls:.1f} deg"
                result["qa"]["ok"] = True
                result["qa"]["reasons"] = []
            else:
                return _reject()  # text rows genuinely tilt: real rejection

        import math as _math
        import statistics as _stats

        _angles = [ln.get("angle", 0.0) for ln in lines if abs(ln.get("angle", 0.0)) < 15]
        _tilt = _stats.median(_angles) if _angles else 0.0
        _shear = _math.tan(_math.radians(_tilt))
        boxes = assign_fields(lines, shear=_shear)
        photo, photo_source = ground_photo_box(image_path)
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
    # Which backend produced the photo box on the model path; None on gold mode.
    result["photo_box_source"] = locals().get("photo_source")
    return result
