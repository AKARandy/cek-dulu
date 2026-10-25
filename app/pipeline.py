"""End-to-end pipeline: QA -> detect -> crop+OCR -> parse -> validate.

`gold=True` swaps model detection for the generator's ground-truth boxes and
stubs OCR with the generator's ground-truth text — the weight-free path used
by CI. Production path (`gold=False`) is identical except boxes/text come
from the models.
"""

from __future__ import annotations

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


def run(
    image_path: str | Path,
    *,
    gold_entry: dict | None = None,
    gold_spec: Any = None,
    workdir: str | Path | None = None,
) -> dict:
    """Process one KTP image. Returns the full result dict (JSON-safe)."""
    image_path = Path(image_path)
    workdir = Path(workdir) if workdir else image_path.parent / "work"
    workdir.mkdir(parents=True, exist_ok=True)

    qa = check_document(image_path)
    result: dict[str, Any] = {"image": image_path.name, "qa": asdict(qa)}

    if not qa.ok:
        result["extracted"] = None
        result["validation"] = None
        result["rejected"] = True
        return result

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
        by_bbox = {tuple(ln["bbox"]): ln for ln in lines}
        boxes = assign_fields(lines)
        photo = ground_photo_box(image_path)
        if photo:
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
        v["text"] = strip_label(v.get("text", ""), key)

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
    result["boxes"] = boxes
    return result
