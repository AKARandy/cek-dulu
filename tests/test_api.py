import pytest
from fastapi.testclient import TestClient

from app.api import app


@pytest.fixture()
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_extract_gold_mode_validates(client):
    r = client.post("/ktp/extract?mode=gold&index=0")
    assert r.status_code == 200
    body = r.json()
    assert body["qa"]["ok"] is True
    assert body["validation"]["ok"] is True, body["validation"]["issues"]
    assert len(body["validation"]["province"]) > 3
    assert body["extracted"]["nik"]["text"].isdigit()


def test_extract_gold_mode_cycles_indices(client):
    for i in (1, 5, 19):
        r = client.post(f"/ktp/extract?mode=gold&index={i}")
        assert r.status_code == 200
        assert r.json()["validation"]["ok"] is True


def test_extract_model_mode_requires_upload(client):
    r = client.post("/ktp/extract?mode=model")
    assert r.status_code == 422


def test_redact_gold_mode_returns_image(client):
    r = client.post("/redact?mode=gold&index=0&fields=nik,foto")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert len(r.content) > 10_000


def test_match_requires_two_files(client):
    r = client.post("/match")
    assert r.status_code == 422
