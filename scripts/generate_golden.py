"""Generate the golden synthetic set into data/golden/ (committed to git)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.synthetic_ktp import generate_batch  # noqa: E402

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    rendered = generate_batch(root / "data" / "golden", n=20)
    print(f"generated {len(rendered)} cards + 3 perturbed + ground_truth.json")
