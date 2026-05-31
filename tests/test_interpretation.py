import unittest

import pandas as pd

from surface_returns.interpretation import (
    beta_characteristic_correlations,
    latent_factor_correlations,
    top_abs_attributions,
)


class InterpretationTests(unittest.TestCase):
    def test_top_abs_attributions(self):
        attr = pd.DataFrame(
            {
                "feature": ["a", "a", "b", "b"],
                "integrated_gradient": [1.0, -3.0, 0.1, 0.2],
            }
        )
        top = top_abs_attributions(attr, top_n=1)
        self.assertEqual(top["feature"].iloc[0], "a")
        self.assertAlmostEqual(float(top["mean_abs_attribution"].iloc[0]), 2.0)

    def test_latent_factor_correlations(self):
        factors = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-31", periods=8, freq="ME"),
                "sdf_factor_01": range(8),
            }
        )
        states = pd.DataFrame(
            {
                "month": pd.date_range("2020-01-31", periods=8, freq="ME"),
                "vix_eom": range(8),
            }
        )
        corr = latent_factor_correlations(factors, states)
        self.assertFalse(corr.empty)
        self.assertAlmostEqual(float(corr["correlation"].iloc[0]), 1.0)

    def test_beta_characteristic_correlations(self):
        dates = pd.date_range("2020-01-31", periods=10, freq="ME")
        assets = pd.DataFrame(
            {
                "date": list(dates) * 20,
                "permno": [item for item in range(20) for _ in dates],
                "sdf_beta_01": [float(item) for item in range(200)],
            }
        )
        chars = assets[["date", "permno"]].copy()
        chars["mean_iv"] = assets["sdf_beta_01"] * 2.0
        corr = beta_characteristic_correlations(assets, chars)
        self.assertFalse(corr.empty)
        self.assertEqual(corr["characteristic"].iloc[0], "mean_iv")


if __name__ == "__main__":
    unittest.main()
