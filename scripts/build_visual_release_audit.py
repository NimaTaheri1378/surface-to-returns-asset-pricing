from __future__ import annotations

import argparse
import csv
import html
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from surface_returns.paths import assert_approved_root, ensure_project_dirs

try:
    from scripts.build_visual_artifact_index import pdf_page_count, png_dimensions
except ModuleNotFoundError:
    from build_visual_artifact_index import pdf_page_count, png_dimensions


CORE_FIGURE_STEMS = [
    "visual_abstract",
    "paper_figure_1_pipeline",
    "paper_figure_2_surface_quality",
    "paper_figure_3_sdf_interpretation",
    "paper_figure_4_return_evidence",
    "paper_figure_5_mechanisms_readiness",
    "visual_evidence_pack",
]

CORE_REPORTS = {
    "visual_report": "outputs/reports/visual_report.html",
    "visual_slide_deck": "outputs/reports/visual_slide_deck.html",
    "visual_pdf_package": "outputs/reports/visual_figure_package.pdf",
    "visual_artifact_index_html": "outputs/reports/visual_artifact_index.html",
    "visual_artifact_index_csv": "outputs/reports/visual_artifact_index.csv",
    "visual_artifact_bundle": "outputs/reports/visual_artifact_bundle.zip",
}

PASS_MANIFESTS = {
    "visual_abstract": "manifests/visual_abstract_manifest.json",
    "paper_figure_package": "manifests/paper_figure_package_manifest.json",
    "visual_pdf_package": "manifests/visual_pdf_package_manifest.json",
    "visual_report": "manifests/visual_report_manifest.json",
    "visual_slide_deck": "manifests/visual_slide_deck_manifest.json",
    "visual_artifact_index": "manifests/visual_artifact_index_manifest.json",
}

READINESS_MANIFEST = "manifests/proposal_readiness_audit_manifest.json"

@dataclass(frozen=True)
class AuditRow:
    group: str
    check: str
    status: str
    detail: str
    path: str = ""


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pass_fail(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def deck_slide_count(path: Path) -> int:
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8", errors="ignore")
    return len(re.findall(r'<section\s+class="slide\b', text))


def bundle_entries(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with zipfile.ZipFile(path) as zf:
        return set(zf.namelist())


def required_bundle_entries() -> set[str]:
    figure_entries = {
        f"outputs/figures/full/{stem}{suffix}"
        for stem in CORE_FIGURE_STEMS
        for suffix in (".png", ".svg")
    }
    report_entries = set(CORE_REPORTS.values()) - {"outputs/reports/visual_artifact_bundle.zip"}
    return figure_entries | report_entries


def audit_core_figures(root: Path) -> list[AuditRow]:
    rows: list[AuditRow] = []
    figures_dir = root / "outputs" / "figures" / "full"
    for stem in CORE_FIGURE_STEMS:
        png = figures_dir / f"{stem}.png"
        svg = figures_dir / f"{stem}.svg"
        width, height = png_dimensions(png) if png.exists() else (None, None)
        dims_ok = bool(width and height and width >= 1200 and height >= 650)
        rows.append(
            AuditRow(
                "core_figures",
                f"{stem} PNG dimensions",
                pass_fail(dims_ok),
                f"{width or 0}x{height or 0}",
                str(png.relative_to(root)),
            )
        )
        svg_ok = svg.exists() and file_size(svg) >= 500
        rows.append(
            AuditRow(
                "core_figures",
                f"{stem} SVG companion",
                pass_fail(svg_ok),
                f"{file_size(svg):,} bytes",
                str(svg.relative_to(root)),
            )
        )
    return rows


def audit_reports(root: Path) -> list[AuditRow]:
    rows: list[AuditRow] = []
    for name, rel in CORE_REPORTS.items():
        path = root / rel
        minimum = 1_000_000 if rel.endswith(".zip") else 1_000
        ok = path.exists() and file_size(path) >= minimum
        rows.append(
            AuditRow(
                "reports",
                f"{name} exists",
                pass_fail(ok),
                f"{file_size(path):,} bytes",
                rel,
            )
        )
    return rows


def audit_manifests(root: Path) -> list[AuditRow]:
    rows: list[AuditRow] = []
    for name, rel in PASS_MANIFESTS.items():
        manifest = read_json(root / rel)
        ok = manifest.get("status") == "PASS"
        rows.append(
            AuditRow(
                "manifests",
                f"{name} status",
                pass_fail(ok),
                str(manifest.get("status", "missing")),
                rel,
            )
        )
    readiness = read_json(root / READINESS_MANIFEST)
    counts = readiness.get("counts", {})
    readiness_ok = (
        readiness.get("status") in {"INCOMPLETE", "PASS_WITH_BOUNDARIES"}
        and int(counts.get("PASS", 0) or 0) >= 14
        and int(counts.get("PASS_WEAK_RESULT", 0) or 0) >= 1
        and int(counts.get("NEGATIVE_RESULT", 0) or 0) >= 3
        and int(counts.get("BLOCKED", 0) or 0) in {0, 1}
    )
    rows.append(
        AuditRow(
            "manifests",
            "readiness boundary documented",
            pass_fail(readiness_ok),
            f"status={readiness.get('status', 'missing')}; counts={counts}",
            READINESS_MANIFEST,
        )
    )
    return rows


def audit_package_contract(root: Path) -> list[AuditRow]:
    rows: list[AuditRow] = []
    pdf_manifest = read_json(root / "manifests" / "visual_pdf_package_manifest.json")
    pdf_path = root / "outputs" / "reports" / "visual_figure_package.pdf"
    actual_pages = pdf_page_count(pdf_path) if pdf_path.exists() else None
    rows.append(
        AuditRow(
            "package_contract",
            "PDF has expected pages",
            pass_fail(pdf_manifest.get("pages") == 9 and actual_pages == 9),
            f"manifest={pdf_manifest.get('pages')}; actual={actual_pages}",
            "outputs/reports/visual_figure_package.pdf",
        )
    )

    deck_manifest = read_json(root / "manifests" / "visual_slide_deck_manifest.json")
    deck_path = root / "outputs" / "reports" / "visual_slide_deck.html"
    actual_slides = deck_slide_count(deck_path)
    rows.append(
        AuditRow(
            "package_contract",
            "deck has expected slides",
            pass_fail(deck_manifest.get("slides") == 9 and actual_slides == 9),
            f"manifest={deck_manifest.get('slides')}; actual={actual_slides}",
            "outputs/reports/visual_slide_deck.html",
        )
    )

    report_manifest = read_json(root / "manifests" / "visual_report_manifest.json")
    report_figures = set(report_manifest.get("figures_available", []))
    expected_figures = {f"{stem}.png" for stem in CORE_FIGURE_STEMS}
    missing = sorted(expected_figures - report_figures)
    rows.append(
        AuditRow(
            "package_contract",
            "report links core figures",
            pass_fail(not missing),
            "missing=" + ",".join(missing) if missing else "all core figures present",
            "manifests/visual_report_manifest.json",
        )
    )

    index_manifest = read_json(root / "manifests" / "visual_artifact_index_manifest.json")
    artifact_count = int(index_manifest.get("artifact_count", 0) or 0)
    zip_sha = str(index_manifest.get("zip_sha256", ""))
    zip_bytes = int(index_manifest.get("zip_bytes", 0) or 0)
    index_ok = artifact_count >= 40 and len(zip_sha) == 64 and zip_bytes >= 1_000_000
    rows.append(
        AuditRow(
            "package_contract",
            "artifact index is substantial",
            pass_fail(index_ok),
            f"artifacts={artifact_count}; zip_bytes={zip_bytes:,}; sha_len={len(zip_sha)}",
            "manifests/visual_artifact_index_manifest.json",
        )
    )

    entries = bundle_entries(root / "outputs" / "reports" / "visual_artifact_bundle.zip")
    required = required_bundle_entries()
    missing_entries = sorted(required - entries)
    rows.append(
        AuditRow(
            "package_contract",
            "artifact bundle contains handoff files",
            pass_fail(not missing_entries),
            "missing=" + ",".join(missing_entries[:6]) if missing_entries else f"{len(entries)} entries",
            "outputs/reports/visual_artifact_bundle.zip",
        )
    )
    return rows


def audit_public_release_guardrails(root: Path) -> list[AuditRow]:
    gitignore = root / ".gitignore"
    text = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    required_ignores = [
        ".env.*",
        "data/processed/",
        "logs/",
        "outputs/figures/",
        "outputs/reports/",
        "outputs/models/",
    ]
    missing = [item for item in required_ignores if item not in text]
    return [
        AuditRow(
            "public_release",
            "generated/private paths ignored",
            pass_fail(not missing),
            "missing=" + ",".join(missing) if missing else "required ignore patterns present",
            ".gitignore",
        )
    ]


def run_audit(root: Path) -> tuple[str, list[AuditRow]]:
    rows = [
        *audit_core_figures(root),
        *audit_reports(root),
        *audit_manifests(root),
        *audit_package_contract(root),
        *audit_public_release_guardrails(root),
    ]
    status = "PASS" if all(row.status == "PASS" for row in rows) else "FAIL"
    return status, rows


def write_csv(path: Path, rows: list[AuditRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "check", "status", "detail", "path"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def render_html(status: str, rows: list[AuditRow]) -> str:
    failures = [row for row in rows if row.status != "PASS"]
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(row.group)}</td>"
            f"<td>{html.escape(row.check)}</td>"
            f'<td><span class="badge badge-{html.escape(row.status.lower())}">{html.escape(row.status)}</span></td>'
            f"<td>{html.escape(row.detail)}</td>"
            f"<td><code>{html.escape(row.path)}</code></td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Surface-to-Returns Visual Release Audit</title>
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
    .badge {{ display: inline-block; min-width: 54px; border-radius: 6px; padding: 4px 7px; font-weight: 800; text-align: center; }}
    .badge-pass {{ color: #2f7d32; background: #edf7ed; }}
    .badge-fail {{ color: #c45a42; background: #fff0eb; }}
  </style>
</head>
<body>
  <header>
    <h1>Visual Release Audit: {html.escape(status)}</h1>
    <p>{len(rows)} checks across the core figures, reports, manifests, package contract, bundle contents, and public-release guardrails. Failures: {len(failures)}.</p>
  </header>
  <main>
    <table>
      <thead><tr><th>Group</th><th>Check</th><th>Status</th><th>Detail</th><th>Path</th></tr></thead>
      <tbody>{''.join(table_rows)}</tbody>
    </table>
  </main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-root-guard", action="store_true")
    args = parser.parse_args()

    root = Path.cwd() if args.no_root_guard else assert_approved_root(Path.cwd())
    ensure_project_dirs(root)
    status, rows = run_audit(root)
    reports = root / "outputs" / "reports"
    csv_path = reports / "visual_release_audit.csv"
    html_path = reports / "visual_release_audit.html"
    manifest_path = root / "manifests" / "visual_release_audit_manifest.json"
    write_csv(csv_path, rows)
    html_path.write_text(render_html(status, rows), encoding="utf-8")
    manifest = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": len(rows),
        "failures": sum(row.status != "PASS" for row in rows),
        "csv": str(csv_path.relative_to(root)),
        "html": str(html_path.relative_to(root)),
        "rows": [row.__dict__ for row in rows],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"visual_release_audit_status={status}")
    print(csv_path.relative_to(root))
    print(html_path.relative_to(root))
    print(manifest_path.relative_to(root))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
