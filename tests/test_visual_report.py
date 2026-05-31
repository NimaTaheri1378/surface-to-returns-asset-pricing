from __future__ import annotations

from pathlib import Path

from scripts.build_visual_report import MetricCard, card_html, fmt_int, fmt_pct, relpath_for_html


def test_formatters_are_report_safe():
    assert fmt_int(12345) == "12,345"
    assert fmt_int("bad") == "n/a"
    assert fmt_pct(0.1234, digits=1) == "12.3%"
    assert fmt_pct(None) == "n/a"


def test_card_html_escapes_text():
    card = MetricCard("A <label>", "1", "safe & sound", "good")

    text = card_html(card)

    assert "A &lt;label&gt;" in text
    assert "safe &amp; sound" in text
    assert "metric-good" in text


def test_relpath_for_html_uses_forward_slashes():
    report = Path("outputs/reports/visual_report.html")
    target = Path("outputs/figures/full/example.png")

    assert relpath_for_html(report, target) == "../figures/full/example.png"
