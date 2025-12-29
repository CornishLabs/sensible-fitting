import os
import subprocess
import sys
from pathlib import Path

import pytest


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
REPO_ROOT = EXAMPLES_DIR.parent
# Only run numbered examples (01_*.py, 02_*.py, ...). This keeps helper scripts
# in examples/ from being treated as CI examples.
EXAMPLE_FILES = sorted(p for p in EXAMPLES_DIR.glob("[0-9][0-9]_*.py") if p.is_file())

OPTIONAL_MODULE_HINTS = {
    "matplotlib": "matplotlib",
    "ultranest": "ultranest",
}


def _skip_missing_optional_modules(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for hint, module in OPTIONAL_MODULE_HINTS.items():
        if hint in text:
            pytest.importorskip(module)


@pytest.mark.examples
@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.name)
def test_example_runs(path: Path) -> None:
    _skip_missing_optional_modules(path)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Example failed: {path.name}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
