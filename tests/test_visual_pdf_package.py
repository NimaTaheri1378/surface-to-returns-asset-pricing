from __future__ import annotations

from pathlib import Path

from scripts.build_visual_pdf_package import PdfFigure, available_figures, fmt_int, fmt_pct


def test_available_figures_filters_missing_files(tmp_path):
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "a.png").write_bytes(b"not-a-real-image")

    specs = [
        PdfFigure("A", "a.png", "exists"),
        PdfFigure("B", "b.png", "missing"),
    ]

    assert available_figures(figures, specs) == [specs[0]]


def test_formatters_are_safe():
    assert fmt_int(1000) == "1,000"
    assert fmt_int("x") == "n/a"
    assert fmt_pct(0.125, 1) == "12.5%"
    assert fmt_pct(None) == "n/a"
