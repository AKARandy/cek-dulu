"""NIK (Nomor Induk Kependudukan) structural validation.

The NIK is 16 digits:
    PP KK CC | DDMMYY | SSSS
    - PP KK CC: 6-digit administrative code (province / regency / district)
    - DDMMYY : birth date; for female cardholders the DAY is encoded +40
               (e.g. day 05 -> 45). Dukcapil practice also uses encoded day
               20 (male) / 60 (female) as an administrative placeholder when
               the true day is unknown.
    - SSSS   : 4-digit serial, unique per admin code + birthdate.

Decoding policy (deliberate, tested):
- Impossible calendar dates are REJECTED, never silently clamped.
- Encoded day 20/60 is decoded literally (day 20). Because 20 is also a
  plausible real birthday, cross-field day mismatches on such NIKs FAIL with
  a caveat naming the placeholder convention instead of silently passing.

This module does NOT claim the NIK "exists" in the Dukcapil registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

# Kemendagri province codes (2-digit). Includes the post-2022 Papua splits.
PROVINCE_CODES: dict[str, str] = {
    "11": "ACEH",
    "12": "SUMATERA UTARA",
    "13": "SUMATERA BARAT",
    "14": "RIAU",
    "15": "JAMBI",
    "16": "SUMATERA SELATAN",
    "17": "BENGKULU",
    "18": "LAMPUNG",
    "19": "KEPULAUAN BANGKA BELITUNG",
    "21": "KEPULAUAN RIAU",
    "31": "DKI JAKARTA",
    "32": "JAWA BARAT",
    "33": "JAWA TENGAH",
    "34": "DI YOGYAKARTA",
    "35": "JAWA TIMUR",
    "36": "BANTEN",
    "51": "BALI",
    "52": "NUSA TENGGARA BARAT",
    "53": "NUSA TENGGARA TIMUR",
    "61": "KALIMANTAN BARAT",
    "62": "KALIMANTAN TENGAH",
    "63": "KALIMANTAN SELATAN",
    "64": "KALIMANTAN TIMUR",
    "65": "KALIMANTAN UTARA",
    "71": "SULAWESI UTARA",
    "72": "SULAWESI TENGAH",
    "73": "SULAWESI SELATAN",
    "74": "SULAWESI TENGGARA",
    "75": "GORONTALO",
    "76": "SULAWESI BARAT",
    "81": "MALUKU",
    "82": "MALUKU UTARA",
    "91": "PAPUA",
    "92": "PAPUA BARAT",
    "93": "PAPUA SELATAN",
    "94": "PAPUA TENGAH",
    "95": "PAPUA PEGUNUNGAN",
    "96": "PAPUA BARAT DAYA",
}

FEMALE_OFFSET = 40
# Encoded days that Dukcapil also uses as placeholders for an unknown day.
PLACEHOLDER_DAYS = {20, 60}


@dataclass
class NikCheck:
    """Result of validating one NIK. `issues` empty means structurally valid."""

    nik: str
    ok: bool = True
    province: str | None = None
    birth_date: date | None = None
    gender: str | None = None  # "LAKI-LAKI" | "PEREMPUAN"
    day_unknown: bool = False  # encoded day 20/60: literal day OR placeholder
    issues: list[str] = field(default_factory=list)

    def _fail(self, msg: str) -> None:
        self.ok = False
        self.issues.append(msg)


def _decode_birth_date(ddmmyy: str) -> tuple[date, str, bool] | None:
    """Decode 6-digit DDMMYY into (date, gender, placeholder_day).

    Returns None when the segment is not a valid calendar date. No clamping:
    310285 (Feb 31) and 290223 (Feb 29 in a non-leap year) are rejected.
    """
    try:
        encoded_day = int(ddmmyy[:2])
        month = int(ddmmyy[2:4])
        yy = int(ddmmyy[4:6])
    except ValueError:
        return None

    gender = "LAKI-LAKI"
    day = encoded_day
    if encoded_day > FEMALE_OFFSET:
        day = encoded_day - FEMALE_OFFSET
        gender = "PEREMPUAN"

    if not 1 <= month <= 12 or not 1 <= day <= 31:
        return None

    century = 2000 if yy <= 26 else 1900  # good until 2026-born turn 100 in 2126
    try:
        decoded = date(century + yy, month, day)
    except ValueError:
        return None  # impossible calendar date: reject, never clamp

    return decoded, gender, encoded_day in PLACEHOLDER_DAYS


def validate_nik(nik: str) -> NikCheck:
    """Validate NIK structure. Digits-only, 16 chars, known province, decodable DOB."""
    check = NikCheck(nik=nik)

    cleaned = nik.strip().replace(" ", "").replace("-", "")
    if cleaned != nik:
        check._fail("NIK contains separators/whitespace; expected bare digits")
    nik = cleaned
    check.nik = nik

    if len(nik) != 16:
        check._fail(f"expected 16 digits, got {len(nik)}")
        return check
    if not nik.isdigit():
        check._fail("contains non-digit characters")
        return check

    province_code = nik[:2]
    if province_code not in PROVINCE_CODES:
        check._fail(f"unknown province code {province_code!r}")
    else:
        check.province = PROVINCE_CODES[province_code]

    decoded = _decode_birth_date(nik[6:12])
    if decoded is None:
        check._fail(f"birth date segment {nik[6:12]!r} is not a valid DDMMYY calendar date")
    else:
        check.birth_date, check.gender, check.day_unknown = decoded

    if nik[12:] == "0000":
        check._fail("serial is 0000 (reserved, never issued)")

    return check


def cross_check_fields(
    check: NikCheck,
    *,
    ttl_date: date | None = None,
    jenis_kelamin: str | None = None,
) -> NikCheck:
    """Cross-check decoded NIK facts against the card's stated fields.

    A day mismatch is a failure even when the encoded day is 20/60: those
    values are plausible real birthdays, so a conflict cannot be waved away
    as a placeholder. The caveat in the issue text names the convention.
    """
    if check.birth_date and ttl_date:
        nik_m, nik_d, nik_y = (
            check.birth_date.month,
            check.birth_date.day,
            check.birth_date.year % 100,
        )
        if (nik_d, nik_m, nik_y) != (ttl_date.day, ttl_date.month, ttl_date.year % 100):
            caveat = (
                " (encoded day 20/60 is also used as an unknown-day placeholder, "
                "so verify against the source document)"
                if check.day_unknown
                else ""
            )
            check._fail(
                f"NIK encodes birth date {check.birth_date:%d-%m-%y} "
                f"but TTL field says {ttl_date:%d-%m-%y}{caveat}"
            )
    if check.gender and jenis_kelamin:
        jk = jenis_kelamin.strip().upper().replace(" ", "")
        if jk in ("LAKI-LAKI", "LAKILAKI", "L", "PRIA") and check.gender != "LAKI-LAKI":
            check._fail("NIK encodes female (+40 day) but Jenis Kelamin says male")
        if jk in ("PEREMPUAN", "WANITA", "P", "W") and check.gender != "PEREMPUAN":
            check._fail("NIK encodes male but Jenis Kelamin says female")
    return check
