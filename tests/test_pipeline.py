import json
from datetime import date
from pathlib import Path

from app.parse_fields import (
    clean_nik,
    normalize,
    normalize_gender,
    normalized_distance,
    parse_ttl_date,
    parse_ttl_place,
)
from app.pipeline import run
from app.synthetic_ktp import KtpSpec

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden"


class TestParsing:
    def test_ttl_parsing(self):
        assert parse_ttl_date("BEKASI, 17-07-1985") == date(1985, 7, 17)
        assert parse_ttl_place("BEKASI, 17-07-1985") == "BEKASI"
        assert parse_ttl_date("JAKARTA SELATAN 13.11.1990") == date(1990, 11, 13)
        assert parse_ttl_date("no date here") is None

    def test_ttl_place_ocr_digit_fix(self):
        # B misread as 8 inside an alphabetic place name
        assert parse_ttl_place("LE8AK, 10-09-1976") == "LEBAK"
        assert parse_ttl_place("BEKASI, 17-07-1985") == "BEKASI"  # pure letters untouched

    def test_nik_cleaning(self):
        assert clean_nik("3216 0517 0785 0123") == "3216051707850123"
        assert clean_nik("NIK: 3216051707850123 X") == "3216051707850123"
        assert clean_nik("3216051707850123999") == "3216051707850123"  # truncates extras

    def test_gender_normalization(self):
        assert normalize_gender("laki-laki") == "LAKI-LAKI"
        assert normalize_gender("Perempuan") == "PEREMPUAN"
        assert normalize_gender("L") == "LAKI-LAKI"
        assert normalize_gender("unknown") is None

    def test_normalize_strips_ocr_junk(self):
        assert normalize("budi_santoso  ") == "BUDI SANTOSO"

    def test_normalized_distance(self):
        assert normalized_distance("BUDI SANTOSO", "budi santoso") == 0.0
        assert 0.0 < normalized_distance("BUDI SANTOSO", "BUDI SANT0SO") < 0.2
        assert normalized_distance("AAAA", "BBBB") == 1.0


class TestPipelineGoldMode:
    def _gold(self, i: int = 0):
        if not (GOLDEN / "ground_truth.json").exists():
            return None, None, None
        gt = json.loads((GOLDEN / "ground_truth.json").read_text(encoding="utf-8"))
        entry = [g for g in gt if g["perturb"] is None][i]
        spec = KtpSpec(
            provinsi=entry["spec"]["provinsi"],
            kabupaten=entry["spec"]["kabupaten"],
            nik=entry["spec"]["nik"],
            nama=entry["spec"]["nama"],
            ttl_tempat=entry["spec"]["ttl_tempat"],
            ttl_tanggal=date.fromisoformat(entry["spec"]["ttl_tanggal"]),
            jenis_kelamin=entry["spec"]["jenis_kelamin"],
            gol_darah=entry["spec"]["gol_darah"],
            alamat=entry["spec"]["alamat"],
            rt_rw=entry["spec"]["rt_rw"],
            kel_desa=entry["spec"]["kel_desa"],
            kecamatan=entry["spec"]["kecamatan"],
            agama=entry["spec"]["agama"],
            status_perkawinan=entry["spec"]["status_perkawinan"],
            pekerjaan=entry["spec"]["pekerjaan"],
        )
        return entry, spec, GOLDEN / entry["image"]

    def test_gold_pipeline_passes_validation(self, tmp_path):
        entry, spec, img = self._gold(0)
        if entry is None:
            return
        result = run(img, gold_entry=entry, gold_spec=spec, workdir=tmp_path)
        assert result["qa"]["ok"]
        assert not result["rejected"]
        assert result["validation"]["ok"], result["validation"]["issues"]
        assert result["extracted"]["nik"]["text"] == spec.nik

    def test_perturbed_cards_are_rejected_by_qa(self, tmp_path):
        if not (GOLDEN / "ground_truth.json").exists():
            return
        gt = json.loads((GOLDEN / "ground_truth.json").read_text(encoding="utf-8"))
        for entry in [g for g in gt if g["perturb"]]:
            result = run(GOLDEN / entry["image"], gold_entry=entry, workdir=tmp_path)
            assert result["rejected"], f"{entry['image']} should fail QA"

    def test_cross_field_mismatch_detected_in_gold_mode(self, tmp_path):
        entry, spec, img = self._gold(1)
        if entry is None:
            return
        tampered = KtpSpec(**{**spec.__dict__})
        tampered.nik = "3216051707850123" if spec.nik != "3216051707850123" else "3116054507850123"
        result = run(img, gold_entry=entry, gold_spec=tampered, workdir=tmp_path)
        # NIK no longer matches the TTL/gender rendered on the card.
        assert not result["validation"]["ok"]
        assert result["validation"]["issues"]
