"""CekDulu Gradio demo: upload a (synthetic) KTP, see extraction with
confidence bars, NIK validation verdicts, redaction toggle, and face
similarity - everything the incumbent eKYC vendors hide behind a sales call.

Run:  .venv/Scripts/python demo/app.py   ->  http://127.0.0.1:7860
HF Spaces: this file is the Space entrypoint (CPU basic tier).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from app import pipeline  # noqa: E402
from app.redact import redact  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden"
DEMO_WORK = ROOT / "data" / "demo_work"

FIELD_LABELS = {
    "nik": "NIK",
    "nama": "Nama",
    "ttl": "Tempat/Tgl Lahir",
    "jenis_kelamin": "Jenis Kelamin",
    "gol_darah": "Gol. Darah",
    "alamat": "Alamat",
    "rt_rw": "RT/RW",
    "kel_desa": "Kel/Desa",
    "kecamatan": "Kecamatan",
    "agama": "Agama",
    "status_perkawinan": "Status Perkawinan",
    "pekerjaan": "Pekerjaan",
    "kewarganegaraan": "Kewarganegaraan",
    "berlaku_hingga": "Berlaku Hingga",
}


def _draw_boxes(image_path: str, boxes: dict) -> Image.Image:
    from PIL import ImageFont

    img = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(img)
    font = None
    fonts = (
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    for cand in fonts:
        if Path(cand).exists():
            font = ImageFont.truetype(cand, 13)
            break
    for key, (x0, y0, x1, y1) in boxes.items():
        d.rectangle([x0, y0, x1, y1], outline=(220, 40, 40), width=2)
        # Label chip inside the box, top-left: never collides with the row above.
        label = FIELD_LABELS.get(key, key)
        tw = d.textlength(label, font=font) if font else len(label) * 7
        cx = max(x0 + 1, x1 - tw - 8)  # top-right: values are left-aligned
        d.rectangle([cx, y0 + 1, cx + tw + 6, y0 + 19], fill=(220, 40, 40))
        d.text((cx + 3, y0 + 3), label, fill=(255, 255, 255), font=font)
    return img


def analyze(image, redact_values: bool):
    """Returns exactly two values, matching the btn.click output binding."""
    if image is None:
        return None, "Upload a KTP image (or generate one - synthetic only!)."
    DEMO_WORK.mkdir(parents=True, exist_ok=True)
    result = pipeline.run(image, workdir=DEMO_WORK)
    if result.get("rejected"):
        reasons = "\n".join(f"- {r}" for r in result["qa"]["reasons"])
        return None, f"**Rejected by document QA**\n\n{reasons}"

    annotated = _draw_boxes(image, result["boxes"])

    v = result["validation"]
    verdict = "✅ VALID" if v["ok"] else "❌ ISSUES FOUND"
    lines = [f"### NIK validation: {verdict}"]
    lines.append(f"- Province: **{v['province'] or '-'}**")
    lines.append(
        f"- Decoded birth date: **{v['birth_date'] or '-'}** (gender: {v['gender'] or '-'})"
    )
    if v["issues"]:
        lines.append("\n**Issues:**")
        lines += [f"- ⚠️ {i}" for i in v["issues"]]
    lines.append("\n### Extracted fields")
    for key, label in FIELD_LABELS.items():
        f = result["extracted"].get(key)
        if f is None:
            continue
        conf = f.get("conf", 0) or 0
        bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
        lines.append(f"- `{label}` {bar} {conf:.0%} → **{f['text'] or '?'}**")

    out_img = annotated
    if redact_values:
        out_img = Image.open(
            redact(display, result["boxes"], out_path=DEMO_WORK / "demo_redacted.png")
        )
    return out_img, "\n".join(lines)


def face_tab(a, b, threshold: float):
    if a is None or b is None:
        return "Upload two face crops."
    from app.face import match

    DEMO_WORK.mkdir(parents=True, exist_ok=True)
    res = match(a, b, threshold=threshold)
    gauge = "🟢 MATCH" if res["same_person"] else "🔴 DIFFERENT"
    return (
        f"### {gauge}\n\n"
        f"Similarity: **{res['similarity']:.4f}** (threshold {res['threshold']})\n\n"
        f"_{res.get('note', '')}_"
    )


CSS = ".footer {visibility: hidden}"


def build() -> gr.Blocks:
    with gr.Blocks(title="cek-dulu", theme=gr.themes.Soft(), css=CSS) as demo:
        gr.Markdown(
            "# 🪪 cek-dulu\n"
            "Open eKYC for Indonesian ID cards. **Synthetic documents only** - every "
            "card this repo has ever seen is generated with fake data and a SPECIMEN "
            "watermark. PaddleOCR + Florence-2 on CPU, no GPU, no API keys."
        )
        with gr.Tab("Extract & validate"):
            with gr.Row():
                with gr.Column():
                    inp = gr.Image(type="filepath", label="KTP image")
                    redact_toggle = gr.Checkbox(
                        False, label="Redact NIK / alamat / photo on output"
                    )
                    btn = gr.Button("Analyze", variant="primary")
                with gr.Column():
                    out_img = gr.Image(label="Fields detected")
                    out_md = gr.Markdown()
            btn.click(analyze, [inp, redact_toggle], [out_img, out_md])
        with gr.Tab("Face similarity (experimental)"):
            with gr.Row():
                fa = gr.Image(type="filepath", label="Face A")
                fb = gr.Image(type="filepath", label="Face B")
            thr = gr.Slider(-1, 1, 0.35, step=0.01, label="Threshold (cosine)")
            fbtn = gr.Button("Compare", variant="primary")
            fout = gr.Markdown()
            fbtn.click(face_tab, [fa, fb, thr], fout)
        gr.Markdown(
            "Experimental: face comparison resizes whole images without detection/"
            "alignment. Field accuracy and timings: see the benchmark table in the README."
        )
    return demo


if __name__ == "__main__":
    build().launch()
