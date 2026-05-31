import unittest

import pandas as pd

from surface_returns.analyst_short import (
    aggregate_ibes_estimates,
    aggregate_short_volume,
    merge_ibes_to_panel,
)


class AnalystShortTests(unittest.TestCase):
    def test_aggregate_ibes_estimates(self):
        raw = pd.DataFrame(
            {
                "ticker": ["abc", "ABC"],
                "statpers": ["2020-01-15", "2020-01-31"],
                "measure": ["EPS", "EPS"],
                "fpi": ["1", "1"],
                "numest": [3, 5],
                "numup": [1, 2],
                "numdown": [0, 1],
                "meanest": [2.0, 4.0],
                "stdev": [0.2, 0.4],
            }
        )
        out = aggregate_ibes_estimates(raw)
        self.assertEqual(len(out), 1)
        self.assertEqual(out["ticker"].iloc[0], "ABC")
        self.assertEqual(float(out["ibes_analyst_coverage"].iloc[0]), 5.0)
        self.assertAlmostEqual(float(out["ibes_forecast_dispersion"].iloc[0]), 0.1)

    def test_merge_ibes_to_panel_uses_link_dates(self):
        panel = pd.DataFrame({"permno": [10], "date": [pd.Timestamp("2020-01-31")]})
        links = pd.DataFrame(
            {
                "permno": [10],
                "ticker": ["ABC"],
                "sdate": [pd.Timestamp("2019-01-01")],
                "edate": [pd.Timestamp("2021-01-01")],
                "score": [1],
            }
        )
        ibes = pd.DataFrame(
            {
                "ticker": ["ABC"],
                "date": [pd.Timestamp("2020-01-01")],
                "ibes_analyst_coverage": [7],
            }
        )
        merged = merge_ibes_to_panel(panel, ibes, links)
        self.assertEqual(float(merged["ibes_analyst_coverage"].iloc[0]), 7.0)

    def test_aggregate_short_volume(self):
        raw = pd.DataFrame(
            {
                "date": ["2020-01-02", "2020-01-03"],
                "symbol": ["ABC", "ABC"],
                "short_q": [10, 20],
                "total_q": [100, 100],
                "shortexempt_q": [1, 1],
            }
        )
        out = aggregate_short_volume(raw)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out["regsho_short_share"].iloc[0]), 0.15)
        self.assertAlmostEqual(float(out["regsho_short_exempt_share"].iloc[0]), 0.01)


if __name__ == "__main__":
    unittest.main()
