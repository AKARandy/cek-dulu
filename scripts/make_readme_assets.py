"""Generate README image assets from the real system (reproducible, no screen capture).

Outputs to docs/img/:
- overlay.png   : detection boxes drawn by the actual pipeline on a golden card
- redact_before.png / redact_after.png : the redaction toggle pair
- demo.gif      : card -> overlay -> validation verdict -> redaction, as frames
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from app import pipeline  # noqa: E402
from app.redact import redact  # noqa: E402
from demo.app import _draw_boxes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "data" / "golden"
IMG = ROOT / "docs" / "img"
WORK = ROOT / "data" / "assets_work"


def main() -> None:
    import json

    IMG.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    gt = json.loads((GOLDEN / "ground_truth.json").read_text(encoding="utf-8"))
    entry = next(g for g in gt if g["perturb"] is None and len(g["spec"]["nama"]) > 8)
    img_path = GOLDEN / entry["image"]

    result = pipeline.run(img_path, workdir=WORK)
    assert not result.get("rejected")

    # 1. detection overlay
    overlay = _draw_boxes(str(img_path), result["boxes"])
    overlay.save(IMG / "overlay.png")

    # 2. redaction pair
    before = Image.open(img_path)
    before.save(IMG / "redact_before.png")
    redact(img_path, result["boxes"], out_path=IMG / "redact_after.png")

    # 3. demo GIF: original -> overlay -> redacted (with pauses)
    redacted = Image.open(IMG / "redact_after.png")
    frames = []
    for im, n in ((before, 40), (overlay, 55), (redacted, 55)):
        frames += [im] * n
    frames[0].save(
        IMG / "demo.gif",
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        optimize=True,
    )

    nik = result["validation"]
    print("assets written to", IMG)
    print("validation ok:", nik["ok"], "| province:", nik["province"])


if __name__ == "__main__":
    main()
