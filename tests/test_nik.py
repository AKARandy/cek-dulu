from datetime import date

import pytest

from validators.nik import PROVINCE_CODES, cross_check_fields, validate_nik


def _make_nik(prov="32", ddmmyy="170785", serial="0123") -> str:
    return f"{prov}1605{ddmmyy}{serial}"


class TestStructure:
    def test_valid_male_nik_passes(self):
        c = validate_nik("3216051707850123")
        assert c.ok, c.issues
        assert c.province == "JAWA BARAT"
        assert c.gender == "LAKI-LAKI"
        assert c.birth_date == date(1985, 7, 17)

    def test_valid_female_nik_plus40(self):
        c = validate_nik("3116054507850123")  # day 45 -> female, day 5
        assert c.ok, c.issues
        assert c.gender == "PEREMPUAN"
        assert c.birth_date == date(1985, 7, 5)

    def test_unknown_day_female_60_decodes_literally(self):
        # 60 = female day 20 under the +40 convention. Decoded literally;
        # placeholder ambiguity is surfaced via day_unknown, not silent passes.
        c = validate_nik("3116056007850123")
        assert c.ok, c.issues
        assert c.gender == "PEREMPUAN"
        assert c.birth_date == date(1985, 7, 20)
        assert c.day_unknown is True

    def test_encoded_day_20_male_is_literal_and_flagged(self):
        c = validate_nik("3116052007850123")
        assert c.ok, c.issues
        assert c.birth_date == date(1985, 7, 20)
        assert c.day_unknown is True

    def test_impossible_calendar_dates_rejected(self):
        for bad in (
            "310285",
            "290223",
            "310498",
            "009985",
        ):  # Feb 31 / Feb 29 non-leap / Apr 31 / day 00
            c = validate_nik(_make_nik(ddmmyy=bad))
            assert not c.ok, f"{bad} should be rejected"
            assert any("DDMMYY" in i for i in c.issues)

    def test_too_short(self):
        c = validate_nik("3216051707")
        assert not c.ok
        assert any("16 digits" in i for i in c.issues)

    def test_non_digits(self):
        c = validate_nik("321605170785012X")
        assert not c.ok
        assert any("non-digit" in i for i in c.issues)

    def test_unknown_province(self):
        c = validate_nik("9916051707850123")
        assert not c.ok
        assert any("province" in i for i in c.issues)

    def test_bad_month(self):
        c = validate_nik("3216051713850123")  # month 13
        assert not c.ok
        assert any("DDMMYY" in i for i in c.issues)

    def test_reserved_serial_zero(self):
        c = validate_nik("3216051707850000")
        assert not c.ok
        assert any("0000" in i for i in c.issues)

    def test_all_38_province_codes_known(self):
        # 34 provinces pre-2022 + Papua splits; codes must be unique + 2-digit.
        assert len(PROVINCE_CODES) == 38
        assert len(set(PROVINCE_CODES)) == 38
        assert all(k.isdigit() and len(k) == 2 for k in PROVINCE_CODES)
        assert PROVINCE_CODES["91"] == "PAPUA"


class TestCrossField:
    def test_ttl_match_passes(self):
        c = validate_nik(_make_nik(ddmmyy="170785"))
        c = cross_check_fields(c, ttl_date=date(1985, 7, 17), jenis_kelamin="LAKI-LAKI")
        assert c.ok, c.issues

    def test_ttl_mismatch_fails(self):
        c = validate_nik(_make_nik(ddmmyy="170785"))
        c = cross_check_fields(c, ttl_date=date(1990, 7, 17))
        assert not c.ok
        assert any("TTL" in i for i in c.issues)

    def test_gender_mismatch_fails(self):
        c = validate_nik(_make_nik(ddmmyy="170785"))  # male
        c = cross_check_fields(c, ttl_date=date(1985, 7, 17), jenis_kelamin="PEREMPUAN")
        assert not c.ok
        assert any("Jenis Kelamin" in i for i in c.issues)

    def test_female_nik_male_field_fails(self):
        c = validate_nik(_make_nik(ddmmyy="450785"))  # female
        c = cross_check_fields(c, ttl_date=date(1985, 7, 5), jenis_kelamin="LAKI-LAKI")
        assert not c.ok

    def test_day_placeholder_mismatch_fails_with_caveat(self):
        # Encoded day 20 vs TTL day 17: fails, and names the placeholder
        # convention instead of silently passing (audit finding 6).
        c = validate_nik(_make_nik(ddmmyy="200785"))
        c = cross_check_fields(c, ttl_date=date(1985, 7, 17), jenis_kelamin="LAKI-LAKI")
        assert not c.ok
        assert any("placeholder" in i for i in c.issues)

    def test_day_placeholder_matching_ttl_passes(self):
        c = validate_nik(_make_nik(ddmmyy="200785"))
        c = cross_check_fields(c, ttl_date=date(1985, 7, 20), jenis_kelamin="LAKI-LAKI")
        assert c.ok, c.issues


@pytest.mark.parametrize(
    "ddmmyy,expected_day,expected_gender",
    [("010101", 1, "LAKI-LAKI"), ("410101", 1, "PEREMPUAN"), ("710199", 31, "PEREMPUAN")],
)
def test_day_decoding(ddmmyy, expected_day, expected_gender):
    c = validate_nik(_make_nik(ddmmyy=ddmmyy))
    assert c.ok, c.issues
    assert c.birth_date.day == expected_day
    assert c.gender == expected_gender
