"""Prove the CI test dependency set is complete on a clean environment.

Creates a fresh venv, installs ONLY what ci.yml installs, regenerates the
golden set, and runs the full weight-free test suite. If collection or any
import fails, the CI job would fail the same way.
"""

import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CI_DEPS = [
    "ruff",
    "pytest",
    "httpx",
    "fastapi",
    "python-multipart",
    "pillow",
    "numpy",
    "gradio",
    "opencv-python-headless",
]


def run(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def main() -> None:
    venv_dir = ROOT / ".venv_ciproof"
    venv.create(venv_dir, with_pip=True)
    if sys.platform == "win32":
        py = str(venv_dir / "Scripts" / "python.exe")
    else:
        py = str(venv_dir / "bin" / "python")

    run([py, "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    run([py, "-m", "pip", "install", "--quiet", *CI_DEPS])
    run([py, "scripts/generate_golden.py"], cwd=ROOT)
    run([py, "-m", "pytest", "tests/", "-q"], cwd=ROOT)
    print("CI DEPENDENCY PROOF: PASSED")


if __name__ == "__main__":
    main()
