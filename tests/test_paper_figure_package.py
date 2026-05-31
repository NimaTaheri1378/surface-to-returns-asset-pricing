from __future__ import annotations

import pandas as pd

from scripts.build_paper_figure_package import fmt_int, fmt_pct, pretty_label, wealth_index


def test_pretty_label_makes_report_labels_readable():
    assert pretty_label("proposal_taq_net_ls") == "Proposal Taq Net Ls"


def test_formatters_handle_missing_values():
    assert fmt_int(1200) == "1,200"
    assert fmt_int("bad") == "n/a"
    assert fmt_pct(1.0) == "100.0%"
    assert fmt_pct(None) == "n/a"


def test_wealth_index_compounds_returns():
    frame = pd.DataFrame({"ret": [0.10, -0.10]})

    assert wealth_index(frame, "ret").round(4).tolist() == [1.1, 0.99]
