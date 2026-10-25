"""Benchmark cek-dulu on the golden set with REAL models (no gold/stub paths).

Writes docs/benchmark.json (raw) and prints a markdown summary.
Run:  .venv/Scripts/python scripts/benchmark.py [--limit N]
Timings are wall-clock on this machine's CPU; state your hardware in the README.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.parse_fields import normalized_distance  # noqa: E402
from app.pipeline import run  # noqa: E402
from app.qa import check_document  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden"
OUT = ROOT / "docs" / "benchmark.json"

TEXT_KEYS = [
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
]


def spec_truth(spec: dict) -> dict[str, str]:
    from datetime import date as _date

    ttl = _date.fromisoformat(spec["ttl_tanggal"])
    return {
        "nik": spec["nik"],
        "nama": spec["nama"],
        "ttl": f"{spec['ttl_tempat'].upper()}, {ttl:%d-%m-%Y}",
        "jenis_kelamin": spec["jenis_kelamin"],
        "gol_darah": spec["gol_darah"],
        "alamat": spec["alamat"],
        "rt_rw": spec["rt_rw"],
        "kel_desa": spec["kel_desa"],
        "kecamatan": spec["kecamatan"],
        "agama": spec["agama"],
        "status_perkawinan": spec["status_perkawinan"],
        "pekerjaan": spec["pekerjaan"],
        "kewarganegaraan": spec["kewarganegaraan"],
    }


def main(limit: int | None = None) -> None:
    gt = json.loads((GOLDEN / "ground_truth.json").read_text(encoding="utf-8"))
    clean = [g for g in gt if g["perturb"] is None][:limit]
    perturbed = [g for g in gt if g["perturb"]]

    exact = {k: 0 for k in TEXT_KEYS}
    total = {k: 0 for k in TEXT_KEYS}
    dists = {k: [] for k in TEXT_KEYS}
    timings = []
    validations_ok = 0
    photo_sources: list[str] = []
    rows = []

    for entry in clean:
        img = GOLDEN / entry["image"]
        t0 = time.time()
        result = run(img, workdir=ROOT / "data" / "bench_work")
        dt = time.time() - t0
        timings.append(dt)
        validations_ok += bool(result["validation"]["ok"])
        photo_sources.append(result.get("photo_box_source") or "gold")
        truth = spec_truth(entry["spec"])
        row = {
            "image": entry["image"],
            "seconds": round(dt, 2),
            "validation_ok": result["validation"]["ok"],
            "photo_box_source": result.get("photo_box_source"),
            "fields": {},
        }
        for key, want in truth.items():
            got = result["extracted"].get(key, {}).get("text", "")
            total[key] += 1
            d = normalized_distance(got, want)
            dists[key].append(d)
            if got.strip().upper() == want.strip().upper():
                exact[key] += 1
            row["fields"][key] = {"got": got, "want": want, "distance": round(d, 3)}
        rows.append(row)
        ok = "OK" if result["validation"]["ok"] else "FAIL"
        print(f"{entry['image']}: {dt:.1f}s validation={ok}")

    qa_rejects = {}
    for entry in perturbed:
        report = check_document(GOLDEN / entry["image"])
        qa_rejects[entry["perturb"]] = {"rejected": not report.ok, "reasons": report.reasons}

    summary = {
        "ran_at": datetime.now(UTC).isoformat(),
        "mode": "real models (PaddleOCR 2.7.3), CPU; photo box provenance recorded per card",
        "cards": len(clean),
        "validation_pass_rate": round(validations_ok / len(clean), 3),
        "photo_box_provenance": {s: photo_sources.count(s) for s in sorted(set(photo_sources))},
        "seconds_per_card": {
            "mean": round(sum(timings) / len(timings), 2),
            "min": round(min(timings), 2),
            "max": round(max(timings), 2),
        },
        "qa_rejection_on_perturbed": qa_rejects,
        "field_accuracy": {
            k: {
                "exact": exact[k],
                "of": total[k],
                "exact_rate": round(exact[k] / total[k], 3) if total[k] else None,
                "mean_normalized_distance": round(sum(dists[k]) / len(dists[k]), 3)
                if dists[k]
                else None,
            }
            for k in TEXT_KEYS
        },
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print("\n| field | exact | of | rate | mean dist |")
    print("|---|---|---|---|---|")
    for k in TEXT_KEYS:
        fa = summary["field_accuracy"][k]
        dist = fa["mean_normalized_distance"]
        print(f"| {k} | {fa['exact']} | {fa['of']} | {fa['exact_rate']} | {dist} |")
    spc = summary["seconds_per_card"]
    print(f"\nvalidation pass: {summary['validation_pass_rate']}, mean {spc['mean']}s/card")
    print(f"photo box provenance: {summary['photo_box_provenance']}")
    print(f"raw: {OUT}")


if __name__ == "__main__":
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    main(limit)
