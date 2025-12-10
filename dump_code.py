#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(".").resolve()
OUT = Path("repo_dump.txt")

INCLUDE = {".py", ".toml", ".md"}
EXCLUDE_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", "dist", "build"}
EXCLUDE_FILES_SUFFIX = {".pyc"}

def should_skip(path: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in path.parts):
        return True
    if path.suffix in EXCLUDE_FILES_SUFFIX:
        return True
    return False

files = []
for p in ROOT.rglob("*"):
    if not p.is_file():
        continue
    if should_skip(p):
        continue
    if p.suffix not in INCLUDE:
        continue
    files.append(p)

files.sort()

with OUT.open("w", encoding="utf-8") as f:
    for p in files:
        rel = p.relative_to(ROOT)
        f.write(f"\n\n# === {rel} ===\n\n")
        try:
            f.write(p.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            f.write("<binary or non-utf8 file skipped>\n")

print(f"Wrote {OUT} with {len(files)} files.")
