from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import struct
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from surface_returns.paths import assert_approved_root, ensure_project_dirs


INCLUDE_REPORTS = [
    "visual_report.html",
    "visual_slide_deck.html",
    "visual_figure_package.pdf",
]
INCLUDE_MANIFESTS = [
    "visual_abstract_manifest.json",
    "paper_figure_package_manifest.json",
    "visual_pdf_package_manifest.json",
    "visual_report_manifest.json",
    "visual_slide_deck_manifest.json",
]
EXCLUDE_NAMES = {
    "visual_report_screenshot.png",
    "visual_slide_deck_screenshot.png",
    "visual_artifact_bundle.zip",
}


@dataclass(frozen=True)
class Artifact:
    path: str
    kind: str
    bytes: int
    sha256: str
    width: int | None = None
    height: int | None = None
    pages: int | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_dimensions(path: Path) -> tuple[int | None, int | None]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None, None
    return struct.unpack(">II", header[16:24])


def svg_dimensions(path: Path) -> tuple[int | None, int | None]:
    text = path.read_text(encoding="utf-8", errors="ignore")[:1000]
    width = re.search(r'\bwidth="([0-9.]+)pt"', text)
    height = re.search(r'\bheight="([0-9.]+)pt"', text)
    if not width or not height:
        return None, None
    return int(float(width.group(1))), int(float(height.group(1)))


def pdf_page_count(path: Path) -> int | None:
    text = path.read_bytes().decode("latin-1", errors="ignore")
    count = len(re.findall(r"/Type\s*/Page\b", text))
    return count or None


def classify(path: Path) -> str:
    if path.suffix.lower() in {".png", ".svg"}:
        if path.name.startswith("paper_figure_"):
            return "paper_figure"
        if path.name.startswith("visual_"):
            return "visual_summary"
        return "diagnostic_figure"
    if path.suffix.lower() == ".pdf":
        return "pdf_package"
    if path.suffix.lower() == ".html":
        return "html_report"
    if path.suffix.lower() == ".json":
        return "manifest"
    return "other"


def collect_artifacts(root: Path) -> list[Path]:
    figures = root / "outputs" / "figures" / "full"
    reports = root / "outputs" / "reports"
    manifests = root / "manifests"
    paths: list[Path] = []
    if figures.exists():
        paths.extend(path for path in sorted(figures.iterdir()) if path.is_file() and path.suffix.lower() in {".png", ".svg"} and path.name not in EXCLUDE_NAMES)
    paths.extend(reports / name for name in INCLUDE_REPORTS if (reports / name).exists())
    paths.extend(manifests / name for name in INCLUDE_MANIFESTS if (manifests / name).exists())
    return paths


def artifact_record(root: Path, path: Path) -> Artifact:
    width = height = pages = None
    suffix = path.suffix.lower()
    if suffix == ".png":
        width, height = png_dimensions(path)
    elif suffix == ".svg":
        width, height = svg_dimensions(path)
    elif suffix == ".pdf":
        pages = pdf_page_count(path)
    return Artifact(
        path=str(path.relative_to(root)).replace("\\", "/"),
        kind=classify(path),
        bytes=path.stat().st_size,
        sha256=sha256_file(path),
        width=width,
        height=height,
        pages=pages,
    )


def write_csv(path: Path, artifacts: list[Artifact]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "kind", "bytes", "sha256", "width", "height", "pages"])
        writer.writeheader()
        for item in artifacts:
            writer.writerow(item.__dict__)


def render_html(artifacts: list[Artifact]) -> str:
    rows = []
    for item in artifacts:
        dims = ""
        if item.width and item.height:
            dims = f"{item.width} x {item.height}"
        elif item.pages:
            dims = f"{item.pages} pages"
        rows.append(
            "<tr>"
            f"<td>{html.escape(item.kind)}</td>"
            f"<td>{html.escape(item.path)}</td>"
            f"<td>{item.bytes:,}</td>"
            f"<td>{html.escape(dims)}</td>"
            f"<td><code>{html.escape(item.sha256[:16])}</code></td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Surface-to-Returns Visual Artifact Index</title>
  <style>
    body {{ margin: 0; font-family: "Aptos", "Segoe UI", Arial, sans-serif; color: #1f2933; background: #fff; }}
    header {{ padding: 38px min(6vw, 72px) 22px; border-bottom: 1px solid #d8dee6; }}
    main {{ padding: 24px min(6vw, 72px) 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 44px; letter-spacing: 0; }}
    p {{ margin: 0; color: #5b6573; font-size: 17px; line-height: 1.45; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 22px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #d8dee6; padding: 10px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f6f8fb; font-size: 12px; text-transform: uppercase; color: #5b6573; }}
    code {{ font-family: Consolas, monospace; }}
  </style>
</head>
<body>
  <header>
    <h1>Visual Artifact Index</h1>
    <p>Inventory of generated figures, reports, manifests, checksums, dimensions, and PDF pages.</p>
  </header>
  <main>
    <p>{len(artifacts)} artifacts indexed. Full SHA256 hashes are in the CSV and JSON manifest.</p>
    <table>
      <thead><tr><th>Kind</th><th>Path</th><th>Bytes</th><th>Dimensions</th><th>SHA256 prefix</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </main>
</body>
</html>
"""


def write_zip(root: Path, artifacts: list[Artifact], zip_path: Path, extra_paths: list[Path] | None = None) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in artifacts:
            source = root / item.path
            if source.resolve() == zip_path.resolve():
                continue
            zf.write(source, item.path)
        for source in extra_paths or []:
            if source.exists() and source.resolve() != zip_path.resolve():
                zf.write(source, str(source.relative_to(root)).replace("\\", "/"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-root-guard", action="store_true")
    args = parser.parse_args()

    root = Path.cwd() if args.no_root_guard else assert_approved_root(Path.cwd())
    ensure_project_dirs(root)
    reports = root / "outputs" / "reports"
    paths = collect_artifacts(root)
    artifacts = [artifact_record(root, path) for path in paths]

    csv_path = reports / "visual_artifact_index.csv"
    html_path = reports / "visual_artifact_index.html"
    zip_path = reports / "visual_artifact_bundle.zip"
    write_csv(csv_path, artifacts)
    html_path.write_text(render_html(artifacts), encoding="utf-8")
    write_zip(root, artifacts, zip_path, extra_paths=[csv_path, html_path])

    manifest = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(artifacts),
        "csv": str(csv_path.relative_to(root)),
        "html": str(html_path.relative_to(root)),
        "zip": str(zip_path.relative_to(root)),
        "zip_sha256": sha256_file(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "artifacts": [item.__dict__ for item in artifacts],
    }
    manifest_path = root / "manifests" / "visual_artifact_index_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("visual_artifact_index_status=PASS")
    print(csv_path.relative_to(root))
    print(html_path.relative_to(root))
    print(zip_path.relative_to(root))
    print(manifest_path.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
