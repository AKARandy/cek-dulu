"""Regression tests for Codex audit 2 (2026-10-25).

1. /redact leaves no req_* dirs behind on invalid uploads, pipeline
   exceptions, render exceptions, or successful delivery.
2. Gradio analyze() arity matches the btn.click binding, checked against the
   live Blocks config, not just the function shape.
3. Overlapping requests (barrier-synchronized at the model boundary) each
   return their own input's result and clean up.
"""

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import api
from app.api import app

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden"


@pytest.fixture()
def client():
    return TestClient(app)


def _golden_bytes(i: int = 0) -> bytes:
    gt = json.loads((GOLDEN / "ground_truth.json").read_text(encoding="utf-8"))
    entry = [g for g in gt if g["perturb"] is None][i]
    return (GOLDEN / entry["image"]).read_bytes()


def _req_dirs(work: Path) -> list[Path]:
    return list(work.glob("req_*")) if work.exists() else []


class TestRedactCleanup:
    def test_invalid_upload_leaves_no_request_dir(self, client, monkeypatch, tmp_path):
        work = tmp_path / "work"
        monkeypatch.setattr(api, "WORK", work)
        r = client.post(
            "/redact?mode=model",
            files={"image": ("bad.png", b"not an image at all", "image/png")},
        )
        assert r.status_code == 422
        assert _req_dirs(work) == []

    def test_pipeline_exception_leaves_no_request_dir(self, monkeypatch, tmp_path):
        work = tmp_path / "work"
        monkeypatch.setattr(api, "WORK", work)

        def boom(*a, **k):
            raise RuntimeError("model exploded")

        monkeypatch.setattr(api.pipeline, "run", boom)
        client = TestClient(app, raise_server_exceptions=False)
        r = client.post(
            "/redact?mode=model",
            files={"image": ("card.png", _golden_bytes(0), "image/png")},
        )
        assert r.status_code == 500
        assert _req_dirs(work) == []

    def test_gold_render_exception_leaves_no_request_dir(self, client, monkeypatch, tmp_path):
        work = tmp_path / "work"
        monkeypatch.setattr(api, "WORK", work)

        def boom(*a, **k):
            raise RuntimeError("render exploded")

        monkeypatch.setattr(api, "redact", boom)
        with pytest.raises(RuntimeError):
            client.post("/redact?mode=gold&index=0")
        assert _req_dirs(work) == []

    def test_successful_redact_cleans_after_delivery(self, client, monkeypatch, tmp_path):
        work = tmp_path / "work"
        monkeypatch.setattr(api, "WORK", work)
        gt = json.loads((GOLDEN / "ground_truth.json").read_text(encoding="utf-8"))
        clean = [g for g in gt if g["perturb"] is None][0]
        b = clean["bboxes"]

        def fake_lines(p):
            def line(key, text):
                x0, y0, x1, y1 = b[key]
                return {"text": text, "bbox": [x0, y0, x1, y1], "conf": 0.99}

            return [
                line("nik", clean["spec"]["nik"]),
                line("nama", clean["spec"]["nama"]),
                line("ttl", "PLACE, 01-01-1990"),
                line("jenis_kelamin", "LAKI-LAKI"),
            ]

        monkeypatch.setattr("app.detect.ocr_lines", fake_lines)
        monkeypatch.setattr(
            "app.detect.ground_photo_box", lambda p: ([610, 100, 810, 350], "layout_prior")
        )
        r = client.post(
            "/redact?mode=model&fields=nik",
            files={"image": ("card.png", _golden_bytes(0), "image/png")},
        )
        assert r.status_code == 200
        assert len(r.content) > 10_000
        assert _req_dirs(work) == []  # BackgroundTask ran after the response


class TestOverlappingRequests:
    def test_barrier_synchronized_requests_stay_isolated(self, monkeypatch, tmp_path):
        work = tmp_path / "work"
        monkeypatch.setattr(api, "WORK", work)
        gt = json.loads((GOLDEN / "ground_truth.json").read_text(encoding="utf-8"))
        clean = [g for g in gt if g["perturb"] is None][:2]

        barrier = threading.Barrier(2, timeout=30)

        def fake_lines(p):
            # Only the saved upload carries card content; ignore the extra
            # files later pipeline stages OCR (nik strip, deskewed copy).
            if Path(p).name != "upload.png":
                return []
            # Identify the card by the upload's own bytes, not call order:
            # each request must get ITS input's lines even mid-flight together.
            digest = hashlib.sha256(Path(p).read_bytes()).hexdigest()
            card = next(
                e
                for e, b in enumerate(_golden_bytes(e) for e in range(2))
                if hashlib.sha256(b).hexdigest() == digest
            )
            entry = clean[card]
            spec, b = entry["spec"], entry["bboxes"]
            barrier.wait()  # hold both requests inside the model boundary

            def line(key, text):
                x0, y0, x1, y1 = b[key]
                return {"text": text, "bbox": [x0, y0, x1, y1], "conf": 0.99}

            return [
                line("nik", spec["nik"]),
                line("nama", spec["nama"]),
                line("ttl", f"{spec['ttl_tempat']}, {spec['ttl_tanggal']}"),
                line("jenis_kelamin", spec["jenis_kelamin"]),
            ]

        import types as _types

        monkeypatch.setattr(
            "app.preprocess.preprocess",
            lambda p, o: (
                Path(p),
                _types.SimpleNamespace(
                    as_dict=lambda: {
                        "method": "test_passthrough",
                        "rotated_by": 0.0,
                        "photometric": False,
                    }
                ),
            ),
        )
        monkeypatch.setattr("app.detect.ocr_lines", fake_lines)
        monkeypatch.setattr(
            "app.detect.ground_photo_box", lambda p: ([610, 100, 810, 350], "layout_prior")
        )

        def post(i: int):
            with TestClient(app) as c:
                r = c.post(
                    "/ktp/extract?mode=model",
                    files={"image": (f"card{i}.png", _golden_bytes(i), "image/png")},
                )
                return i, r.status_code, r.json()["extracted"]["nik"]["text"]

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = sorted(pool.map(post, (0, 1)))

        for i, status, nik in results:
            assert status == 200
            assert nik == clean[i]["spec"]["nik"]
        assert _req_dirs(work) == []


class TestGradioWiring:
    def test_analyze_arity_matches_click_binding(self, monkeypatch, tmp_path):
        import demo.app as d

        canned = {
            "rejected": False,
            "boxes": {},
            "validation": {
                "ok": True,
                "province": "TEST",
                "birth_date": None,
                "gender": None,
                "issues": [],
            },
            "extracted": {},
        }
        monkeypatch.setattr(d.pipeline, "run", lambda *a, **k: dict(canned))
        monkeypatch.setattr(d, "DEMO_WORK", tmp_path)

        gt = json.loads((GOLDEN / "ground_truth.json").read_text(encoding="utf-8"))
        first = [g for g in gt if g["perturb"] is None][0]
        # The callback contract: two outputs, two return values.
        result = d.analyze(str(GOLDEN / first["image"]), False)
        assert isinstance(result, tuple) and len(result) == 2

        blocks = d.build()
        # Blocks.fns is a list in some Gradio versions, a dict in others.
        fns_list = list(blocks.fns.values()) if isinstance(blocks.fns, dict) else list(blocks.fns)
        wired = [fn for fn in fns_list if getattr(fn, "outputs", None)]
        assert wired, "no wired callbacks found"
        click_outputs = {len(fn.outputs) for fn in wired}
        assert 2 in click_outputs, f"analyze binding not found in outputs sizes {click_outputs}"
