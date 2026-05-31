from __future__ import annotations

import struct
import zipfile

from scripts.build_visual_artifact_index import (
    Artifact,
    classify,
    png_dimensions,
    sha256_file,
    write_zip,
)


def tiny_png_bytes(width: int = 3, height: int = 2) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_len = struct.pack(">I", 13)
    ihdr_type = b"IHDR"
    ihdr_data = struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    crc = b"\x00\x00\x00\x00"
    return signature + ihdr_len + ihdr_type + ihdr_data + crc


def test_png_dimensions_reads_ihdr(tmp_path):
    path = tmp_path / "x.png"
    path.write_bytes(tiny_png_bytes(7, 5))

    assert png_dimensions(path) == (7, 5)


def test_sha256_file_is_stable(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("abc", encoding="utf-8")

    assert sha256_file(path) == sha256_file(path)
    assert len(sha256_file(path)) == 64


def test_classify_visual_outputs(tmp_path):
    assert classify(tmp_path / "paper_figure_1_pipeline.png") == "paper_figure"
    assert classify(tmp_path / "visual_report.html") == "html_report"
    assert classify(tmp_path / "visual_figure_package.pdf") == "pdf_package"


def test_write_zip_uses_relative_artifact_paths(tmp_path):
    source = tmp_path / "outputs" / "reports" / "visual_report.html"
    source.parent.mkdir(parents=True)
    source.write_text("ok", encoding="utf-8")
    zip_path = tmp_path / "outputs" / "reports" / "bundle.zip"
    artifact = Artifact(
        path="outputs/reports/visual_report.html",
        kind="html_report",
        bytes=2,
        sha256="0" * 64,
    )

    extra = tmp_path / "outputs" / "reports" / "visual_artifact_index.csv"
    extra.write_text("index", encoding="utf-8")

    write_zip(tmp_path, [artifact], zip_path, extra_paths=[extra])

    with zipfile.ZipFile(zip_path) as zf:
        assert zf.namelist() == [
            "outputs/reports/visual_report.html",
            "outputs/reports/visual_artifact_index.csv",
        ]
