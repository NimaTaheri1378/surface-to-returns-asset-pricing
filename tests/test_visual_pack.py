from __future__ import annotations

import pandas as pd

from scripts.build_visual_evidence_pack import safe_float, wealth_index


def test_wealth_index_compounds_missing_returns_as_zero():
    frame = pd.DataFrame({"ret": [0.10, None, -0.10]})

    wealth = wealth_index(frame, "ret")

    assert wealth.round(6).tolist() == [1.1, 1.1, 0.99]

def test_safe_float_rejects_bad_values():
    assert safe_float("bad", default=3.0) == 3.0
    assert safe_float("1.25") == 1.25
