"""CekDulu API: QA, extraction+validation, face similarity, redaction.

Two modes:
- `mode=gold` (no upload): runs the pipeline on the committed synthetic golden
  set using generator ground truth. Weight-free; used by CI, the demo's
  "try it" button, and any smoke test.
- `mode=model` (upload): the real path — PaddleOCR lines + layout priors +
  Florence-2 photo grounding.

Security posture (audit-driven): client filenames are never used as paths;
every request gets its own temporary directory under WORK; uploads are
size-capped and decode-verified; FileResponse artifacts outlive the handler
via background cleanup.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from app import pipeline
from app.redact import redact
from app.synthetic_ktp import KtpSpec

app = FastAPI(
    title="cek-dulu",
    version="0.2.0",
    description="Open eKYC for Indonesian ID cards (synthetic data only).",
)

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden"
WORK = ROOT / "data" / "api_work"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _golden(index: int) -> tuple[dict, KtpSpec, Path]:
    gt_path = GOLDEN / "ground_truth.json"
    if not gt_path.exists():
        raise HTTPException(503, "golden set missing; run scripts/generate_golden.py")
    entries = [g for g in json.loads(gt_path.read_text(encoding="utf-8")) if g["perturb"] is None]
    entry = entries[index % len(entries)]
    s = entry["spec"]
    spec = KtpSpec(
        provinsi=s["provinsi"],
        kabupaten=s["kabupaten"],
        nik=s["nik"],
        nama=s["nama"],
        ttl_tempat=s["ttl_tempat"],
        ttl_tanggal=date.fromisoformat(s["ttl_tanggal"]),
        jenis_kelamin=s["jenis_kelamin"],
        gol_darah=s["gol_darah"],
        alamat=s["alamat"],
        rt_rw=s["rt_rw"],
        kel_desa=s["kel_desa"],
        kecamatan=s["kecamatan"],
        agama=s["agama"],
        status_perkawinan=s["status_perkawinan"],
        pekerjaan=s["pekerjaan"],
    )
    return entry, spec, GOLDEN / entry["image"]


def _save_upload(upload: UploadFile, tmpdir: Path, name: str = "upload.png") -> Path:
    """Store an upload under a server-controlled path, size-capped and verified.

    Decoding is verified in memory BEFORE anything touches the disk, so an
    invalid upload leaves no file behind. The client filename is deliberately
    never used as (part of) a path.
    """
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(422, "upload exceeds 10 MB limit")
    if not data:
        raise HTTPException(422, "upload is empty")
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            im.verify()
    except Exception as exc:  # noqa: BLE001 - any decode failure is a 422
        raise HTTPException(422, f"upload is not a decodable image: {exc}") from exc
    dest = tmpdir / name
    dest.write_bytes(data)
    return dest


def _new_request_dir() -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="req_", dir=WORK))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "cek-dulu", "version": app.version}


@app.post("/ktp/extract")
def ktp_extract(
    image: UploadFile | None = File(None),
    mode: str = Query("gold", pattern="^(gold|model)$"),
    index: int = Query(0, ge=0),
) -> JSONResponse:
    if mode == "gold":
        entry, spec, img = _golden(index)
        return JSONResponse(pipeline.run(img, gold_entry=entry, gold_spec=spec, workdir=WORK))
    if image is None:
        raise HTTPException(422, "mode=model requires an image upload")
    req = _new_request_dir()
    try:
        dest = _save_upload(image, req)
        result = pipeline.run(dest, workdir=req / "work")
    finally:
        shutil.rmtree(req, ignore_errors=True)
    return JSONResponse(result)


@app.post("/redact")
def redact_image(
    image: UploadFile | None = File(None),
    mode: str = Query("gold", pattern="^(gold|model)$"),
    index: int = Query(0, ge=0),
    fields: str = Query("nik,alamat,foto"),
) -> FileResponse:
    wanted = tuple(f.strip() for f in fields.split(",") if f.strip())

    if mode == "gold":
        entry, _, img = _golden(index)
        boxes = entry.get("bboxes") or {}
        req = _new_request_dir()
        try:
            out = redact(img, boxes, fields=wanted, out_path=req / "redacted.png")
            skipped = [f for f in wanted if f not in boxes]
        except Exception:
            shutil.rmtree(req, ignore_errors=True)
            raise
        return FileResponse(
            out,
            media_type="image/png",
            headers={"X-Skipped-Fields": ",".join(skipped)},
            background=BackgroundTask(shutil.rmtree, req, ignore_errors=True),
        )

    if image is None:
        raise HTTPException(422, "mode=model requires an image upload")
    req = _new_request_dir()
    try:
        dest = _save_upload(image, req)
        result = pipeline.run(dest, workdir=req / "work")
        if result.get("rejected"):
            raise HTTPException(422, f"image rejected by QA: {result['qa']['reasons']}")
        skipped = [f for f in wanted if f not in (result.get("boxes") or {})]
        out = redact(dest, result["boxes"], fields=wanted, out_path=req / "redacted.png")
    except Exception:
        shutil.rmtree(req, ignore_errors=True)
        raise
    return FileResponse(
        out,
        media_type="image/png",
        headers={"X-Skipped-Fields": ",".join(skipped)},
        background=BackgroundTask(shutil.rmtree, req, ignore_errors=True),
    )


@app.post("/match")
def match_faces(
    a: UploadFile = File(...), b: UploadFile = File(...), threshold: float = 0.35
) -> dict:
    if not -1.0 <= threshold <= 1.0:
        raise HTTPException(422, "threshold must be within [-1, 1] (cosine similarity)")
    from app.face import match as face_match

    req = _new_request_dir()
    try:
        pa = _save_upload(a, req, "face_a.png")
        pb = _save_upload(b, req, "face_b.png")
        result = face_match(pa, pb, threshold=threshold)
    finally:
        shutil.rmtree(req, ignore_errors=True)
    result["note"] = (
        "experimental crop-similarity only: whole images are resized and embedded "
        "without face detection or alignment; this is not identity verification"
    )
    return result
