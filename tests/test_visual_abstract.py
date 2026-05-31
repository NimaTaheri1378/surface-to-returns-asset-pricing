from __future__ import annotations

from scripts.build_visual_abstract import fmt_int, fmt_pct, return_evidence_label


def test_formatters_handle_bad_values():
    assert fmt_int(1234567) == "1,234,567"
    assert fmt_int("bad") == "n/a"
    assert fmt_pct(1.0) == "100.0%"
    assert fmt_pct("bad") == "n/a"


def test_return_evidence_label_marks_cost_aware_portfolio():
    label, color = return_evidence_label(
        {"mean_rank_ic": 0.01},
        {"net": {"mean_monthly_return": -0.002}},
    )

    assert label == "cost-aware"
    assert color
