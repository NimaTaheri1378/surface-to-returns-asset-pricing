from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.build_visual_slide_deck import SlideSpec, fmt_int, image_slide, relpath_for_html, status_counts


def test_relpath_for_html_is_portable():
    assert (
        relpath_for_html(
            Path("outputs/reports/visual_slide_deck.html"),
            Path("outputs/figures/full/visual_abstract.png"),
        )
        == "../figures/full/visual_abstract.png"
    )


def test_image_slide_skips_missing_image(tmp_path):
    deck = tmp_path / "outputs" / "reports" / "visual_slide_deck.html"
    figures = tmp_path / "outputs" / "figures" / "full"
    figures.mkdir(parents=True)

    assert image_slide(deck, figures, SlideSpec("Missing", "No image", "missing.png"), 1) == ""


def test_status_and_int_formatters():
    frame = pd.DataFrame({"status": ["PASS", "PASS", "BLOCKED"]})

    assert status_counts(frame) == {"PASS": 2, "BLOCKED": 1}
    assert fmt_int(1234) == "1,234"
    assert fmt_int("bad") == "n/a"
