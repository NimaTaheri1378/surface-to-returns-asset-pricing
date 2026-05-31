import unittest

import pandas as pd

from surface_returns.state_controls import build_state_control_table, merge_state_controls


class StateControlTests(unittest.TestCase):
    def test_build_and_merge_state_controls_by_month(self):
        ff = pd.DataFrame(
            {
                "date": ["2020-01-31", "2020-02-29"],
                "mktrf": [1.0, -2.0],
                "smb": [0.1, 0.2],
                "hml": [0.3, 0.4],
                "rf": [0.01, 0.01],
                "umd": [0.5, 0.6],
            }
        )
        ff5 = pd.DataFrame({"date": ["2020-01-31", "2020-02-29"], "rmw": [0.1, 0.2], "cma": [0.3, 0.4]})
        frb = pd.DataFrame(
            {
                "date": ["2020-01-31", "2020-02-29"],
                "aaa": [3.0, 3.1],
                "baa": [4.0, 4.2],
                "fedfunds": [1.5, 1.4],
                "mswp1": [1.6, 1.5],
                "mswp10": [2.6, 2.4],
            }
        )
        cboe = pd.DataFrame(
            {
                "date": ["2020-01-02", "2020-01-31", "2020-02-28"],
                "vix": [13.0, 15.0, 25.0],
                "vxo": [12.0, 14.0, 24.0],
            }
        )
        state = build_state_control_table(ff, ff5, frb, cboe)
        self.assertIn("vix_eom", state)
        self.assertIn("term_spread_10y_ff", state)
        panel = pd.DataFrame({"date": pd.to_datetime(["2020-01-30", "2020-02-28"]), "permno": [1, 1]})
        merged = merge_state_controls(panel, state)
        self.assertEqual(float(merged.loc[0, "vix_eom"]), 15.0)
        self.assertEqual(float(merged.loc[1, "mktrf"]), -2.0)


if __name__ == "__main__":
    unittest.main()
