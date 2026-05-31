import unittest

import pandas as pd

from surface_returns.surfaces import clean_option_quotes, fixed_surface_grid, surface_features
from surface_returns.svi import (
    SSVIParams,
    SVIParams,
    fit_ssvi_surface,
    fit_svi_slice,
    fit_svi_surface_grid,
    raw_svi_total_variance,
    ssvi_no_arbitrage_bounds,
    ssvi_surface_no_arbitrage_checks,
    ssvi_total_variance,
    svi_static_arbitrage_checks,
)


class SurfaceTests(unittest.TestCase):
    def sample(self):
        return pd.DataFrame(
            {
                "date": ["2022-12-30"] * 4,
                "secid": [1, 1, 1, 1],
                "exdate": ["2023-01-30", "2023-02-28", "2023-01-30", "2023-02-28"],
                "cp_flag": ["C", "C", "P", "P"],
                "best_bid": [1.0, 1.1, 1.2, 1.3],
                "best_offer": [1.2, 1.3, 1.5, 1.6],
                "impl_volatility": [0.25, 0.27, 0.31, 0.34],
                "delta": [0.50, 0.45, -0.50, -0.35],
                "strike_price": [100, 105, 100, 95],
            }
        )

    def test_clean_and_features(self):
        cleaned = clean_option_quotes(self.sample())
        self.assertEqual(len(cleaned), 4)
        feats = surface_features(cleaned)
        self.assertEqual(len(feats), 1)
        self.assertGreater(feats["put_call_iv_spread"].iloc[0], 0)

    def test_fixed_grid(self):
        grid = fixed_surface_grid(self.sample(), maturities=[30, 60], deltas=[0.25, 0.50])
        self.assertEqual(len(grid), 4)
        self.assertTrue(grid["impl_volatility"].notna().all())

    def test_svi_slice_fits_synthetic_smile(self):
        params = SVIParams(a=0.025, b=0.12, rho=-0.35, m=0.0, sigma=0.25)
        k = pd.Series([-0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45])
        dte = 60
        total_variance = raw_svi_total_variance(k, params)
        quotes = pd.DataFrame(
            {
                "date": ["2022-12-30"] * len(k),
                "secid": [1] * len(k),
                "exdate": ["2023-02-28"] * len(k),
                "cp_flag": ["P"] * 5 + ["C"] * 5,
                "best_bid": [1.0] * len(k),
                "best_offer": [1.1] * len(k),
                "impl_volatility": (total_variance / (dte / 365.25)) ** 0.5,
                "delta": [0.1, 0.18, 0.28, 0.38, 0.48, 0.55, 0.65, 0.75, 0.85, 0.92],
                "strike_price": 100 * k.apply(lambda x: 2.718281828459045**x),
            }
        )
        fitted, diagnostics = fit_svi_slice(quotes, min_quotes=8)
        self.assertIsNotNone(fitted)
        self.assertEqual(diagnostics["status"], "PASS")
        self.assertLess(diagnostics["rmse_total_variance"], 0.01)
        checks = svi_static_arbitrage_checks(fitted)
        self.assertTrue(checks["positive_total_variance"])

    def test_svi_surface_grid_calendar_monotone(self):
        base = self.sample()
        richer = pd.concat([base.assign(strike_price=90 + i * 2, delta=-0.9 + i * 0.1) for i in range(18)])
        richer["best_bid"] = 1.0
        richer["best_offer"] = 1.2
        richer["impl_volatility"] = 0.20 + (pd.to_numeric(richer["delta"]).abs() - 0.5).abs() * 0.25
        grid, diagnostics = fit_svi_surface_grid(richer, maturities=[30, 60], deltas=[0.25, 0.50, 0.75], min_quotes=8)
        self.assertFalse(diagnostics.empty)
        self.assertFalse(grid.empty)
        for _delta, group in grid.groupby("target_call_delta"):
            ordered = group.sort_values("target_dte")
            self.assertTrue((ordered["total_variance"].diff().fillna(0) >= -1e-10).all())

    def test_ssvi_bounds(self):
        checks = ssvi_no_arbitrage_bounds(theta=0.04, phi=1.5, rho=-0.4)
        self.assertTrue(checks["ssvi_bounds_pass"])

    def test_ssvi_global_surface_fit_passes_no_arb_checks(self):
        params = SSVIParams(rho=-0.35, eta=1.2, lam=0.25)
        maturities = [30, 60, 120]
        theta = {30: 0.04, 60: 0.07, 120: 0.11}
        k_grid = [-0.45, -0.20, 0.0, 0.20, 0.45]
        rows = []
        for maturity in maturities:
            for idx, k in enumerate(k_grid):
                rows.append(
                    {
                        "date": "2022-12-30",
                        "secid": 1,
                        "target_dte": maturity,
                        "target_call_delta": [0.10, 0.25, 0.50, 0.75, 0.90][idx],
                        "target_log_moneyness": k,
                        "total_variance": ssvi_total_variance(pd.Series([k]), theta[maturity], params)[0],
                    }
                )
        fitted, fitted_grid, diagnostics = fit_ssvi_surface(pd.DataFrame(rows), min_points=12)
        self.assertIsNotNone(fitted)
        self.assertEqual(diagnostics["status"], "PASS")
        self.assertFalse(fitted_grid.empty)
        self.assertLess(diagnostics["rmse_total_variance"], 0.002)
        checks = ssvi_surface_no_arbitrage_checks(fitted["theta_by_maturity"], fitted["params"])
        self.assertTrue(checks["calendar_monotone_grid"])
        self.assertTrue(checks["ssvi_bounds_pass"])


if __name__ == "__main__":
    unittest.main()
