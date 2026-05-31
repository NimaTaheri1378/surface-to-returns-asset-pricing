from __future__ import annotations

import json
import struct
import zipfile

from scripts.build_visual_release_audit import (
    CORE_FIGURE_STEMS,
    bundle_entries,
    deck_slide_count,
    required_bundle_entries,
    run_audit,
)


def fake_png(width: int = 1600, height: int = 900) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_len = struct.pack(">I", 13)
    ihdr_type = b"IHDR"
    ihdr_data = struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    crc = b"\x00\x00\x00\x00"
    return signature + ihdr_len + ihdr_type + ihdr_data + crc + (b"0" * 1200)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_minimal_release_tree(root):
    figures = root / "outputs" / "figures" / "full"
    reports = root / "outputs" / "reports"
    manifests = root / "manifests"
    figures.mkdir(parents=True)
    reports.mkdir(parents=True)
    manifests.mkdir(parents=True)

    for stem in CORE_FIGURE_STEMS:
        (figures / f"{stem}.png").write_bytes(fake_png())
        (figures / f"{stem}.svg").write_text('<svg width="1600pt" height="900pt"></svg>' + (" " * 600), encoding="utf-8")

    (reports / "visual_report.html").write_text("report" * 300, encoding="utf-8")
    (reports / "visual_slide_deck.html").write_text(
        ('<section class="slide"></section>' + (" " * 120)) * 9,
        encoding="utf-8",
    )
    (reports / "visual_figure_package.pdf").write_bytes(b"%PDF\n" + (b"/Type /Page\n" * 9) + (b"0" * 1200))
    (reports / "visual_artifact_index.html").write_text("index" * 300, encoding="utf-8")
    (reports / "visual_artifact_index.csv").write_text("path,kind\n" + ("x,y\n" * 400), encoding="utf-8")

    zip_path = reports / "visual_artifact_bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for rel in sorted(required_bundle_entries()):
            source = root / rel
            if source.exists():
                zf.write(source, rel)
        zf.writestr("padding.bin", b"0" * 1_100_000)

    for name in [
        "visual_abstract",
        "paper_figure_package",
        "visual_pdf_package",
        "visual_report",
        "visual_slide_deck",
        "visual_artifact_index",
    ]:
        write_json(manifests / f"{name}_manifest.json", {"status": "PASS"})

    write_json(
        manifests / "visual_pdf_package_manifest.json",
        {"status": "PASS", "pages": 9},
    )
    write_json(
        manifests / "visual_slide_deck_manifest.json",
        {"status": "PASS", "slides": 9},
    )
    write_json(
        manifests / "visual_report_manifest.json",
        {"status": "PASS", "figures_available": [f"{stem}.png" for stem in CORE_FIGURE_STEMS]},
    )
    write_json(
        manifests / "visual_artifact_index_manifest.json",
        {"status": "PASS", "artifact_count": 51, "zip_sha256": "a" * 64, "zip_bytes": zip_path.stat().st_size},
    )
    write_json(
        manifests / "proposal_readiness_audit_manifest.json",
        {
            "status": "INCOMPLETE",
            "counts": {"PASS": 14, "PASS_WEAK_RESULT": 1, "NEGATIVE_RESULT": 3, "BLOCKED": 1},
            "blocking_or_partial_items": ["External APIs"],
        },
    )
    (root / ".gitignore").write_text(
        ".env.*\ndata/processed/\nlogs/\noutputs/figures/\noutputs/reports/\noutputs/models/\n",
        encoding="utf-8",
    )


def test_deck_slide_count_counts_slide_sections(tmp_path):
    path = tmp_path / "deck.html"
    path.write_text('<section class="slide title-slide"></section><section class="slide image-slide"></section>', encoding="utf-8")

    assert deck_slide_count(path) == 2


def test_bundle_entries_reads_zip_names(tmp_path):
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("a.txt", "a")

    assert bundle_entries(path) == {"a.txt"}


def test_run_audit_passes_on_complete_minimal_release_tree(tmp_path):
    build_minimal_release_tree(tmp_path)

    status, rows = run_audit(tmp_path)

    assert status == "PASS"
    assert rows
    assert {row.status for row in rows} == {"PASS"}
