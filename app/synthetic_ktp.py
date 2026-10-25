"""Synthetic KTP generator.

Renders fake Indonesian ID cards with Pillow. Every value is fake, every image
carries a SPECIMEN watermark, and the exact bounding box of each rendered field
is returned as ground truth — which gives the detection benchmark its gold
annotations for free.

Only synthetic data is ever used in this repo (see PLAN.md hard rules).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from validators.nik import PROVINCE_CODES

# Canonical KTP field order (label on card).
FIELDS = [
    "nik",
    "nama",
    "ttl",
    "jenis_kelamin",
    "gol_darah",
    "alamat",
    "rt_rw",
    "kel_desa",
    "kecamatan",
    "agama",
    "status_perkawinan",
    "pekerjaan",
    "kewarganegaraan",
    "berlaku_hingga",
]

FIRST_NAMES_M = ["BUDI", "AGUS", "RAHMAT", "JOKO", "DIMAS", "FAJAR", "HENDRA", "YOGA"]
FIRST_NAMES_F = ["SITI", "DEWI", "RATNA", "PUTRI", "MEGA", "LARAS", "INTAN", "AYU"]
LAST_NAMES = [
    "SANTOSO",
    "WIBOWO",
    "NUGRAHA",
    "PRATAMA",
    "MAHENDRA",
    "RAMADHAN",
    "KUSUMA",
    "SETIAWAN",
    "ANGGRAINI",
    "HANDAYANI",
]
CITIES = {
    "32": [("BEKASI", "KAB", "32.16", "TARUMAJAYA"), ("BANDUNG", "KOTA", "32.73", "COBLONG")],
    "31": [("JAKARTA SELATAN", "KOTA", "31.71", "TEBET")],
    "35": [("SURABAYA", "KOTA", "35.78", "TAMBAKSARI")],
    "34": [("YOGYAKARTA", "KOTA", "34.71", "GONDOKUSUMAN")],
    "51": [("DENPASAR", "KOTA", "51.71", "DENPASAR SELATAN")],
}
AGAMA = ["ISLAM", "KRISTEN", "KATOLIK", "HINDU", "BUDDHA", "KONGHUCU"]
STATUS = ["BELUM MENIKAH", "MENIKAH", "CERAI HIDUP", "CERAI MATI"]
PEKERJAAN = [
    "PELAJAR/MAHASISWA",
    "KARYAWAN SWASTA",
    "PEGAWAI NEGERI SIPIL",
    "WIRASWASTA",
    "PROGRAMMER",
    "GURU",
    "PETANI",
]
JOBS = ["programmer", "karyawan swasta", "pegawai negeri sipil", "wiraswasta"]

W, H = 850, 540


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    for cand in (f"C:/Windows/Fonts/{name}", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(cand).exists():
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


@dataclass
class KtpSpec:
    """All values rendered onto one synthetic card."""

    provinsi: str
    kabupaten: str
    nik: str
    nama: str
    ttl_tempat: str
    ttl_tanggal: date
    jenis_kelamin: str
    gol_darah: str
    alamat: str
    rt_rw: str
    kel_desa: str
    kecamatan: str
    agama: str
    status_perkawinan: str
    pekerjaan: str
    kewarganegaraan: str = "WNI"
    berlaku_hingga: str = "SEUMUR HIDUP"


@dataclass
class RenderedKtp:
    spec: KtpSpec
    image_path: str
    bboxes: dict[str, list[int]] = field(default_factory=dict)  # x0,y0,x1,y1


def _random_nik(rng: random.Random, prov: str, birth: date, female: bool) -> str:
    # Always encodes the true day. The day-20/60 placeholder convention is a
    # real-data administrative artifact; synthetic cards have no unknown days,
    # so placeholder cases are covered by unit tests instead.
    day = birth.day + (40 if female else 0)
    ddmmyy = f"{day:02d}{birth.month:02d}{birth.year % 100:02d}"
    serial = f"{rng.randint(1, 9999):04d}"
    regency = rng.choice(["16", "73", "71", "78"] if prov == "32" else ["01", "02", "05"])
    district = rng.choice(["01", "05", "09", "12"])
    return f"{prov}{regency}{district}{ddmmyy}{serial}"


def make_spec(rng: random.Random, seed: int) -> KtpSpec:
    prov = rng.choice(["32", "31", "35", "34", "51"])
    prov_name = PROVINCE_CODES[prov]
    city, kind, _, kec = CITIES[prov][rng.randrange(len(CITIES[prov]))]
    female = rng.random() < 0.5
    first = rng.choice(FIRST_NAMES_F if female else FIRST_NAMES_M)
    nama = f"{first} {rng.choice(LAST_NAMES)}"
    birth = date(rng.randint(1975, 2004), rng.randint(1, 12), rng.randint(1, 28))
    gol = rng.choice(["A", "B", "AB", "O", "-"])
    street = rng.choice(["MERDEKA", "SUDIRMAN", "GAJAH MADA", "KARTINI", "DIPONEGORO"])
    return KtpSpec(
        provinsi=f"PROVINSI {prov_name}",
        kabupaten=f"{kind} {city}"
        if kind != "KOTA" or city.startswith("JAKARTA")
        else f"KOTA {city}",
        nik=_random_nik(rng, prov, birth, female),
        nama=nama,
        ttl_tempat=city,
        ttl_tanggal=birth,
        jenis_kelamin="PEREMPUAN" if female else "LAKI-LAKI",
        gol_darah=gol,
        alamat=f"JL. {street} NO {rng.randint(1, 120)}",
        rt_rw=f"{rng.randint(1, 15):03d}/{rng.randint(1, 15):03d}",
        kel_desa=rng.choice(["SUKAMAJU", "MEKARSARI", "CIBINONG", "KALIBARU", "TEBET"]).upper(),
        kecamatan=kec,
        agama=rng.choice(AGAMA),
        status_perkawinan=rng.choice(STATUS),
        pekerjaan=rng.choice(PEKERJAAN),
    )


def render(spec: KtpSpec, out_path: Path, *, perturb: str | None = None) -> RenderedKtp:
    """Render one card. `perturb` in {None, 'blur', 'glare', 'skew'} for QA tests."""
    img = Image.new("RGB", (W, H), (232, 236, 240))
    d = ImageDraw.Draw(img)

    f_head = _font(22, bold=True)
    f_label = _font(13)
    f_value = _font(14)
    f_nik = _font(20, bold=True)
    f_ktp = _font(34, bold=True)

    # Header band
    d.rectangle([0, 0, W, 54], fill=(233, 246, 252))
    d.text((28, 8), spec.provinsi, font=f_head, fill=(0, 60, 110))
    d.text((28, 30), spec.kabupaten, font=f_head, fill=(0, 60, 110))
    d.text((610, 14), "KTP", font=f_ktp, fill=(200, 40, 40))
    d.line([0, 54, W, 54], fill=(120, 140, 160), width=2)

    bboxes: dict[str, list[int]] = {}

    def put(key: str, label: str, value: str, x: int, y: int, value_dx: int = 108) -> None:
        d.text((x, y), f"{label}:", font=f_label, fill=(60, 60, 60))
        # Value column: keep the standard offset, but never let long labels
        # (Status Perkawinan, Kewarganegaraan) collide with their values.
        vx = x + max(value_dx, int(d.textlength(f"{label}:", font=f_label)) + 10)
        d.text((vx, y - 2), value, font=f_value, fill=(15, 15, 15))
        tw = d.textlength(value, font=f_value)
        bboxes[key] = [vx - 2, y - 4, int(vx + tw) + 2, y + 18]

    # NIK row
    d.text((28, 66), "NIK", font=f_label, fill=(60, 60, 60))
    d.text((108, 60), spec.nik, font=f_nik, fill=(15, 15, 15))
    bboxes["nik"] = [106, 56, 108 + int(d.textlength(spec.nik, font=f_nik)) + 2, 88]

    # Left column fields
    put("nama", "Nama", spec.nama, 28, 100)
    put(
        "ttl",
        "Tempat/Tgl Lahir",
        f"{spec.ttl_tempat}, {spec.ttl_tanggal:%d-%m-%Y}",
        28,
        124,
        value_dx=108,
    )
    put("jenis_kelamin", "Jenis Kelamin", spec.jenis_kelamin, 28, 148, value_dx=108)
    put("gol_darah", "Gol. Darah", spec.gol_darah, 320, 148, value_dx=76)
    put("alamat", "Alamat", spec.alamat, 28, 172, value_dx=108)
    put("rt_rw", "RT/RW", spec.rt_rw, 28, 196, value_dx=108)
    put("kel_desa", "Kel/Desa", spec.kel_desa, 28, 220, value_dx=108)
    put("kecamatan", "Kecamatan", spec.kecamatan, 28, 244, value_dx=108)
    put("agama", "Agama", spec.agama, 28, 286, value_dx=108)
    put("status_perkawinan", "Status Perkawinan", spec.status_perkawinan, 28, 310, value_dx=108)
    put("pekerjaan", "Pekerjaan", spec.pekerjaan, 28, 334, value_dx=108)
    put("kewarganegaraan", "Kewarganegaraan", spec.kewarganegaraan, 28, 358, value_dx=108)
    put("berlaku_hingga", "Berlaku Hingga", spec.berlaku_hingga, 28, 382, value_dx=108)

    # Photo box (right) with a simple drawn silhouette
    px, py, pw, ph = 610, 100, 200, 250
    d.rectangle([px, py, px + pw, py + ph], fill=(205, 210, 216), outline=(90, 100, 110), width=2)
    d.ellipse([px + 60, py + 40, px + 140, py + 120], fill=(160, 168, 178))  # head
    d.ellipse([px + 40, py + 130, px + 160, py + 250], fill=(150, 158, 170))  # body
    bboxes["foto"] = [px, py, px + pw, py + ph]
    # Signature strip under photo
    d.rectangle([px, py + ph + 14, px + pw, py + ph + 50], outline=(90, 100, 110), width=1)
    d.text((px + 30, py + ph + 24), "specimen", font=_font(18), fill=(130, 130, 130))

    # Bottom band
    d.rectangle([0, H - 26, W, H], fill=(0, 70, 120))
    d.rectangle([0, H - 30, W, H - 26], fill=(230, 140, 30))

    # SPECIMEN watermark
    wm = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    wd = ImageDraw.Draw(wm)
    wd.text(
        (W // 2 - 150, H // 2 - 30), "SPECIMEN", font=_font(44, bold=True), fill=(220, 30, 30, 60)
    )
    img = Image.alpha_composite(img.convert("RGBA"), wm).convert("RGB")
    d = ImageDraw.Draw(img)
    d.text(
        (30, H - 50), "SYNTHETIC SAMPLE - NOT A REAL DOCUMENT", font=_font(12), fill=(150, 30, 30)
    )

    if perturb == "blur":
        from PIL import ImageFilter

        img = img.filter(ImageFilter.GaussianBlur(3))
    elif perturb == "glare":
        # Saturated specular patch, the way real overexposure clips.
        overlay = Image.new("L", (W, H), 0)
        od = ImageDraw.Draw(overlay)
        od.ellipse([350, 120, 620, 300], fill=255)
        img = Image.composite(Image.new("RGB", (W, H), (255, 255, 255)), img, overlay)
    elif perturb == "skew":
        img = img.rotate(7, expand=False, fillcolor=(232, 236, 240))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

    return RenderedKtp(
        spec=spec,
        image_path=str(out_path),
        bboxes=bboxes,
    )


def generate_batch(out_dir: Path, n: int = 20, *, seed0: int = 1) -> list[RenderedKtp]:
    """Generate n clean cards + a few perturbed ones, with ground-truth JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[RenderedKtp] = []
    gt = []
    for i in range(n):
        rng = random.Random(seed0 + i)
        spec = make_spec(rng, seed0 + i)
        path = out_dir / f"ktp_{i:03d}.png"
        r = render(spec, path)
        rendered.append(r)
        gt.append({"image": path.name, "spec": asdict(spec), "bboxes": r.bboxes, "perturb": None})
    # Perturbed variants for QA gates (indices n, n+1, n+2)
    for j, p in enumerate(("blur", "glare", "skew")):
        rng = random.Random(seed0 + 1000 + j)
        spec = make_spec(rng, seed0 + 1000 + j)
        path = out_dir / f"ktp_{p}.png"
        render(spec, path, perturb=p)
        gt.append({"image": path.name, "spec": asdict(spec), "bboxes": {}, "perturb": p})
    (out_dir / "ground_truth.json").write_text(
        json.dumps(gt, indent=1, default=str), encoding="utf-8"
    )
    return rendered
