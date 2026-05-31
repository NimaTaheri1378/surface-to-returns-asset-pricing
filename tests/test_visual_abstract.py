from __future__ import annotations

import pandas as pd

from scripts.build_visual_abstract import fmt_int, fmt_pct, return_evidence_label, status_counts


def test_status_counts_reads_status_column():
    frame = pd.DataFrame({"status": ["PASS", "PASS", "BLOCKED"]})

    assert status_counts(frame) == {"PASS": 2, "BLOCKED": 1}


def test_formatters_handle_bad_values():
    assert fmt_int(1234567) == "1,234,567"
    assert fmt_int("bad") == "n/a"
    assert fmt_pct(1.0) == "100.0%"
    assert fmt_pct("bad") == "n/a"


def test_return_evidence_label_marks_negative_portfolio():
    label, color = return_evidence_label(
        {"mean_rank_ic": 0.01},
        {"net": {"mean_monthly_return": -0.002}},
    )

    assert label == "weak/negative"
    assert color
