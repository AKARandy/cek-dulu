"""Redaction: blur sensitive regions (NIK, address block, photo) on output images."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter

DEFAULT_FIELDS = ("nik", "alamat", "foto")


def redact(
    image_path: str | Path,
    boxes: dict[str, list[int]],
    *,
    fields: tuple[str, ...] = DEFAULT_FIELDS,
    blur_radius: int = 12,
    out_path: str | Path | None = None,
) -> Path:
    """Return a copy of the image with the given field boxes blurred."""
    img = Image.open(image_path).convert("RGB")
    for key in fields:
        if key not in boxes:
            continue
        x0, y0, x1, y1 = boxes[key]
        region = img.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(blur_radius))
        img.paste(region, (x0, y0))
    out_path = (
        Path(out_path)
        if out_path
        else Path(image_path).with_name(
            Path(image_path).stem + "_redacted" + Path(image_path).suffix
        )
    )
    img.save(out_path)
    return out_path
