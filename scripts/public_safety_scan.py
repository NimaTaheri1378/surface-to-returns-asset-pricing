from __future__ import annotations

import argparse
import re
from pathlib import Path

from surface_returns.paths import assert_approved_root


SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|secret|password)[ \t]*[:=][ \t]*['\"]?[A-Za-z0-9_\-]{12,}"),
    re.compile(r"(?i)SEC_EDGAR_USER_AGENT[ \t]*[:=][ \t]*['\"]?.+@.+"),
]

SKIP_DIRS = {
    ".codex",
    ".git",
    "__pycache__",
    ".pytest_cache",
    "data",
    "logs",
    "outputs",
    ".venv",
    "venv",
}

SKIP_FILE_NAMES = {
    ".env",
}

ALLOWED_SECRET_PLACEHOLDERS = {
    "FRED_API_KEY=",
    "BLS_API_KEY=",
    "BEA_API_KEY=",
    "EIA_API_KEY=",
    "SEC_EDGAR_USER_AGENT=",
}


def iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = set(path.relative_to(root).parts)
        if rel_parts & SKIP_DIRS:
            continue
        if path.name in SKIP_FILE_NAMES or path.name.startswith(".env."):
            continue
        yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-manifests", action="store_true")
    parser.add_argument(
        "--allow-any-root",
        action="store_true",
        help="Permit read-only scans in CI checkouts outside the approved research workspaces.",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve() if args.allow_any_root else assert_approved_root(Path.cwd())
    findings = []
    for path in iter_files(root):
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] == "manifests" and args.fail_on_manifests:
            findings.append((str(rel), "manifest file present in public tree"))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                snippet = match.group(0)
                if snippet in ALLOWED_SECRET_PLACEHOLDERS:
                    continue
                findings.append((str(rel), "possible secret pattern"))
                break
    if findings:
        print("public_safety_status=FAIL")
        for rel, reason in findings:
            print(f"{rel}: {reason}")
        return 2
    print("public_safety_status=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
