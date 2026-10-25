import random
from pathlib import Path

from PIL import Image

from app.qa import check_document, estimate_skew, glare_fraction, sharpness_score
from app.synthetic_ktp import make_spec, render


def _cards(tmp_path: Path):
    """One clean card + the three perturbed variants."""
    rng = random.Random(7)
    spec = make_spec(rng, 7)
    paths = {}
    for perturb in (None, "blur", "glare", "skew"):
        name = f"card_{perturb or 'clean'}.png"
        render(spec, tmp_path / name, perturb=perturb)
        paths[perturb or "clean"] = tmp_path / name
    return paths


def test_clean_card_passes_all_gates(tmp_path):
    paths = _cards(tmp_path)
    report = check_document(paths["clean"])
    assert report.ok, report.reasons


def test_blur_card_rejected_for_sharpness(tmp_path):
    paths = _cards(tmp_path)
    report = check_document(paths["blur"])
    assert not report.ok
    assert any(c.name == "sharpness" and not c.passed for c in report.checks)
    # and the reason string names it
    assert any("sharpness" in r for r in report.reasons)


def test_glare_card_rejected_for_glare(tmp_path):
    paths = _cards(tmp_path)
    report = check_document(paths["glare"])
    assert not report.ok
    assert any(c.name == "glare" and not c.passed for c in report.checks)


def test_skew_card_rejected_for_skew(tmp_path):
    paths = _cards(tmp_path)
    report = check_document(paths["skew"])
    skew_check = next(c for c in report.checks if c.name == "skew")
    assert not skew_check.passed  # ~7 degrees vs 4.0 threshold


def test_scores_move_in_the_expected_direction(tmp_path):
    paths = _cards(tmp_path)
    clean = Image.open(paths["clean"])
    assert sharpness_score(Image.open(paths["blur"])) < sharpness_score(clean)
    assert glare_fraction(Image.open(paths["glare"])) > glare_fraction(clean)
    assert abs(estimate_skew(Image.open(paths["skew"]))) > 4.0
