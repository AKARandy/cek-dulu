"""Regression tests for the Codex audit findings (2026-10-25).

Covers: upload path traversal, request isolation, missing-NIK structured
result, fresh-WORK redaction, upload size/decode limits, threshold bounds,
and empty-OCR robustness with the model boundary mocked.
"""

import json
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


def _mock_ocr(monkeypatch, lines_by_path: dict | None = None, lines: list | None = None):
    """Mock the model boundary: OCR lines (canned) + Florence photo (fallback)."""
    if lines_by_path is not None:
        monkeypatch.setattr(
            "app.detect.ocr_lines",
            lambda p: lines_by_path[Path(p).name],
        )
    else:
        import types as _types

    def _identity_preprocess(path, out):
        return Path(path), _types.SimpleNamespace(
            as_dict=lambda: {"method": "test_passthrough", "rotated_by": 0.0, "photometric": False}
        )

    monkeypatch.setattr("app.preprocess.preprocess", _identity_preprocess)
    monkeypatch.setattr("app.detect.ocr_lines", lambda p: lines or [])
    monkeypatch.setattr(
        "app.detect.ground_photo_box", lambda p: ([610, 100, 810, 350], "layout_prior")
    )


class TestPathTraversal:
    def test_traversal_filename_cannot_escape_work(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(api, "WORK", tmp_path / "work")
        _mock_ocr(monkeypatch, lines=[])
        before = {p.name for p in tmp_path.rglob("*")}
        r = client.post(
            "/ktp/extract?mode=model",
            files={"image": ("../../escaped.txt", _golden_bytes(0), "image/png")},
        )
        assert r.status_code == 200
        after = {p.name for p in tmp_path.rglob("*")}
        assert "escaped.txt" not in after
        assert not (tmp_path / "escaped.txt").exists()
        assert not any("escaped" in n for n in after - before)

    def test_absolute_path_filename_is_ignored(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(api, "WORK", tmp_path / "work")
        _mock_ocr(monkeypatch, lines=[])
        r = client.post(
            "/ktp/extract?mode=model",
            files={"image": ("C:/Windows/system32/evil.png", _golden_bytes(0), "image/png")},
        )
        assert r.status_code == 200
        assert not (Path("C:/Windows/system32/evil.png")).exists()


class TestRequestIsolation:
    def test_two_uploads_do_not_share_state(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(api, "WORK", tmp_path / "work")
        gt = json.loads((GOLDEN / "ground_truth.json").read_text(encoding="utf-8"))
        clean = [g for g in gt if g["perturb"] is None]
        calls = {"n": 0}

        def fake_lines(p):
            # Only the saved upload carries card content; ignore the extra
            # files later pipeline stages OCR (nik strip, deskewed copy).
            if Path(p).name != "upload.png":
                return []
            # Uploads get server-controlled names, so identify the card by
            # call order: first upload -> card 0, second -> card 1.
            entry = clean[calls["n"] % 2]
            calls["n"] += 1
            spec = entry["spec"]
            b = entry["bboxes"]

            def line(key, text):
                x0, y0, x1, y1 = b[key]
                return {"text": text, "bbox": [x0, y0, x1, y1], "conf": 0.99}

            return [
                line("nik", spec["nik"]),
                line("nama", spec["nama"]),
                line("ttl", f"{spec['ttl_tempat']}, {spec['ttl_tanggal']}"),
                line("jenis_kelamin", spec["jenis_kelamin"]),
            ]

        _mock_ocr(monkeypatch)
        monkeypatch.setattr("app.detect.ocr_lines", fake_lines)  # per-image canned lines

        r1 = client.post(
            "/ktp/extract?mode=model", files={"image": ("a.png", _golden_bytes(0), "image/png")}
        )
        r2 = client.post(
            "/ktp/extract?mode=model", files={"image": ("b.png", _golden_bytes(1), "image/png")}
        )
        n1 = r1.json()["extracted"]["nik"]["text"]
        n2 = r2.json()["extracted"]["nik"]["text"]
        assert n1 == clean[0]["spec"]["nik"]
        assert n2 == clean[1]["spec"]["nik"]
        assert n1 != n2
        # request dirs cleaned up
        assert not list((tmp_path / "work").glob("req_*"))


class TestMissingNik:
    def test_empty_ocr_gives_structured_validation_failure(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(api, "WORK", tmp_path / "work")
        _mock_ocr(monkeypatch, lines=[])  # QA passes, OCR finds nothing
        r = client.post(
            "/ktp/extract?mode=model", files={"image": ("card.png", _golden_bytes(0), "image/png")}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["qa"]["ok"] is True
        assert "nik" in body["extracted"]  # present, empty - no KeyError
        assert body["extracted"]["nik"]["text"] == ""
        assert body["validation"]["ok"] is False
        assert any("16 digits" in i for i in body["validation"]["issues"])


class TestRedactRobustness:
    def test_redact_model_mode_on_fresh_work_dir(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(api, "WORK", tmp_path / "brand_new_work")  # does not exist yet
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
            "/redact?mode=model&fields=nik,foto",
            files={"image": ("card.png", _golden_bytes(0), "image/png")},
        )
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "image/png"
        assert len(r.content) > 10_000

    def test_skipped_fields_are_reported_not_silent(self, client):
        r = client.post("/redact?mode=gold&index=0&fields=nik,nonexistent_field")
        assert r.status_code == 200
        assert r.headers.get("x-skipped-fields") == "nonexistent_field"


class TestUploadValidation:
    def test_oversized_upload_rejected(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(api, "WORK", tmp_path / "work")
        big = b"\x89PNG\r\n\x1a\n" + b"0" * (10 * 1024 * 1024 + 1)
        r = client.post("/ktp/extract?mode=model", files={"image": ("big.png", big, "image/png")})
        assert r.status_code == 422

    def test_garbage_bytes_rejected(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(api, "WORK", tmp_path / "work")
        r = client.post(
            "/ktp/extract?mode=model", files={"image": ("x.png", b"not an image", "image/png")}
        )
        assert r.status_code == 422


class TestMatchValidation:
    def test_threshold_out_of_range_rejected(self, client):
        for bad in (1.5, -2.0):
            r = client.post(
                "/match",
                files={
                    "a": ("a.png", _golden_bytes(0), "image/png"),
                    "b": ("b.png", _golden_bytes(1), "image/png"),
                },
                params={"threshold": bad},
            )
            assert r.status_code == 422
