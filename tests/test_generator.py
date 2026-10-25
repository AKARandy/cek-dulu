import json
import random
from pathlib import Path

from app.synthetic_ktp import generate_batch, make_spec, render
from validators.nik import cross_check_fields, validate_nik

ROOT = Path(__file__).resolve().parent.parent


def _spec():
    rng = random.Random(42)
    return make_spec(rng, 42)


def test_generated_nik_is_structurally_valid():
    for seed in range(50):
        spec = make_spec(random.Random(seed), seed)
        c = validate_nik(spec.nik)
        assert c.ok, f"seed {seed}: {c.issues} (nik={spec.nik})"
        assert spec.provinsi == f"PROVINSI {c.province}"


def test_generated_nik_cross_checks_with_fields():
    for seed in range(50):
        spec = make_spec(random.Random(seed), seed)
        c = validate_nik(spec.nik)
        c = cross_check_fields(c, ttl_date=spec.ttl_tanggal, jenis_kelamin=spec.jenis_kelamin)
        assert c.ok, (
            f"seed {seed}: {c.issues} (nik={spec.nik}, ttl={spec.ttl_tanggal}, "
            f"jk={spec.jenis_kelamin})"
        )


def test_render_produces_image_and_bboxes(tmp_path):
    r = render(_spec(), tmp_path / "ktp.png")
    assert Path(r.image_path).exists()
    img_size = (tmp_path / "ktp.png").stat().st_size
    assert img_size > 10_000  # non-trivial image
    for key in ("nik", "nama", "ttl", "alamat", "foto"):
        assert key in r.bboxes, f"missing bbox {key}"
        x0, y0, x1, y1 = r.bboxes[key]
        assert 0 <= x0 < x1 <= 850 and 0 <= y0 < y1 <= 540


def test_batch_generation_writes_ground_truth(tmp_path):
    generate_batch(tmp_path / "out", n=5)
    gt = json.loads((tmp_path / "out" / "ground_truth.json").read_text(encoding="utf-8"))
    assert len(gt) == 5 + 3  # clean + perturbed
    assert {g["perturb"] for g in gt if g["perturb"]} == {"blur", "glare", "skew"}
    clean = [g for g in gt if g["perturb"] is None]
    assert all(len(g["bboxes"]) >= 14 for g in clean)


def test_golden_set_exists_and_is_valid():
    gt_path = ROOT / "data" / "golden" / "ground_truth.json"
    if not gt_path.exists():  # generated later by scripts/generate_golden.py
        return
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    assert len(gt) >= 23
    for g in gt:
        if g["perturb"] is None:
            assert validate_nik(g["spec"]["nik"]).ok
