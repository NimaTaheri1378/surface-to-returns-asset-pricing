import unittest

import pandas as pd

from surface_returns.inference import (
    decile_returns_from_predictions,
    factor_alpha_table,
    grs_test,
    moving_block_bootstrap_summary,
    normalize_factor_units,
)


class InferenceTests(unittest.TestCase):
    def test_normalize_factor_units(self):
        frame = pd.DataFrame({"mktrf": [1.0, -2.0], "rf": [0.01, 0.02]})
        out = normalize_factor_units(frame, ["mktrf", "rf"])
        self.assertAlmostEqual(float(out["mktrf"].iloc[0]), 0.01)
        self.assertAlmostEqual(float(out["rf"].iloc[0]), 0.01)

    def test_decile_returns(self):
        preds = pd.DataFrame(
            {
                "date": ["2020-01-31"] * 100,
                "permno": list(range(100)),
                "pred": list(range(100)),
                "next_ret": [idx / 1000 for idx in range(100)],
            }
        )
        deciles = decile_returns_from_predictions(preds)
        self.assertEqual(len(deciles), 1)
        self.assertGreater(float(deciles["decile_ls"].iloc[0]), 0)

    def test_factor_alpha_and_grs(self):
        dates = pd.date_range("2020-01-31", periods=36, freq="ME").to_period("M").to_timestamp()
        returns = pd.DataFrame(
            {
                "date": dates,
                "p1": [0.01] * 36,
                "p2": [0.02] * 36,
            }
        )
        factors = pd.DataFrame(
            {
                "date": dates,
                "mktrf": [0.0] * 36,
                "smb": [0.0] * 36,
                "hml": [0.0] * 36,
                "rf": [0.0] * 36,
            }
        )
        alpha = factor_alpha_table(returns, factors, ["p1"], ["mktrf", "smb", "hml"], lags=3)
        self.assertEqual(alpha["status"].iloc[0], "PASS")
        self.assertAlmostEqual(float(alpha["alpha_monthly"].iloc[0]), 0.01)
        grs = grs_test(returns, factors, ["p1", "p2"], ["mktrf", "smb", "hml"])
        self.assertIn(grs["status"], {"PASS", "SKIPPED_TOO_FEW_OBSERVATIONS"})

    def test_block_bootstrap(self):
        summary = moving_block_bootstrap_summary(pd.Series([0.01, -0.02, 0.03, 0.01] * 10), n_boot=100)
        self.assertEqual(summary["status"], "PASS")
        self.assertIn("mean_monthly_ci_low", summary)


if __name__ == "__main__":
    unittest.main()
