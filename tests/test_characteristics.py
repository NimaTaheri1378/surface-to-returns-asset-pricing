import unittest

import pandas as pd

from surface_returns.characteristics import (
    collapse_daily_risk_to_panel_months,
    compute_compustat_annual_characteristics,
    compute_crsp_monthly_characteristics,
    compute_daily_risk_characteristics,
    merge_compustat_characteristics,
)


class CharacteristicTests(unittest.TestCase):
    def test_crsp_monthly_characteristics_are_lag_safe(self):
        monthly = pd.DataFrame(
            {
                "permno": [1] * 14,
                "date": pd.date_range("2020-01-31", periods=14, freq="ME"),
                "ret": [0.01] * 14,
                "prc": [10.0] * 14,
                "shrout": [1000.0] * 14,
                "vol": [100.0] * 14,
            }
        )
        chars = compute_crsp_monthly_characteristics(monthly)
        self.assertIn("momentum_12_2", chars)
        self.assertIn("momentum_6_1", chars)
        self.assertIn("amihud_illiq_12m", chars)
        self.assertAlmostEqual(float(chars["market_equity"].iloc[0]), 10_000_000.0)
        self.assertTrue(chars["momentum_12_2"].iloc[-1] > 0)
        self.assertTrue(chars["momentum_6_1"].iloc[-1] > 0)
        self.assertTrue(chars["amihud_illiq_12m"].iloc[-1] > 0)

    def test_compustat_characteristics_and_availability(self):
        funda = pd.DataFrame(
            {
                "gvkey": ["001", "001"],
                "datadate": ["2020-12-31", "2021-12-31"],
                "fyear": [2020, 2021],
                "at": [100.0, 120.0],
                "seq": [60.0, 70.0],
                "ceq": [55.0, 65.0],
                "txditc": [5.0, 6.0],
                "pstk": [1.0, 1.0],
                "sale": [200.0, 220.0],
                "cogs": [120.0, 130.0],
                "xsga": [30.0, 33.0],
                "xint": [2.0, 2.0],
                "ni": [10.0, 12.0],
                "capx": [8.0, 9.0],
                "act": [40.0, 44.0],
                "che": [5.0, 6.0],
                "lct": [30.0, 32.0],
                "dlc": [4.0, 5.0],
                "dltt": [20.0, 21.0],
                "txp": [2.0, 2.0],
                "dp": [3.0, 3.0],
                "csho": [10.0, 10.0],
                "prcc_f": [15.0, 16.0],
            }
        )
        chars = compute_compustat_annual_characteristics(funda)
        self.assertIn("book_equity", chars)
        self.assertIn("asset_growth", chars)
        self.assertIn("sales_growth", chars)
        self.assertIn("cash_at", chars)
        self.assertIn("earnings_to_price", chars)
        self.assertEqual(str(chars["available_date"].iloc[0].date()), "2021-06-30")
        self.assertAlmostEqual(float(chars["investment"].iloc[1]), 0.2)
        self.assertAlmostEqual(float(chars["asset_growth"].iloc[1]), 0.2)
        self.assertAlmostEqual(float(chars["sales_growth"].iloc[1]), 0.1)
        self.assertAlmostEqual(float(chars["cash_at"].iloc[1]), 0.05)
        self.assertAlmostEqual(float(chars["earnings_to_price"].iloc[1]), 12.0 / 160.0)

    def test_compustat_merge_uses_latest_available_record(self):
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(["2021-05-31", "2021-07-31"]),
                "permno": [10, 10],
                "secid": [1, 1],
            }
        )
        comp = pd.DataFrame(
            {
                "gvkey": ["001"],
                "datadate": pd.to_datetime(["2020-12-31"]),
                "available_date": pd.to_datetime(["2021-06-30"]),
                "book_equity": [100.0],
            }
        )
        ccm = pd.DataFrame(
            {
                "gvkey": ["001"],
                "permno": [10],
                "linkdt": pd.to_datetime(["2000-01-01"]),
                "linkenddt": [pd.NaT],
                "linktype": ["LU"],
                "linkprim": ["P"],
            }
        )
        merged = merge_compustat_characteristics(panel, comp, ccm)
        early = merged.loc[merged["date"].eq(pd.Timestamp("2021-05-31")), "book_equity"]
        late = merged.loc[merged["date"].eq(pd.Timestamp("2021-07-31")), "book_equity"]
        self.assertTrue(early.isna().all())
        self.assertAlmostEqual(float(late.iloc[0]), 100.0)

    def test_daily_risk_uses_percent_factor_units_and_panel_months(self):
        dates = pd.bdate_range("2020-01-01", periods=90)
        market = pd.Series([0.1 + i * 0.001 for i in range(len(dates))])
        daily = pd.DataFrame(
            {
                "permno": [10] * len(dates),
                "date": dates,
                "ret": market / 100.0 * 1.5,
            }
        )
        factors = pd.DataFrame({"date": dates, "mktrf": market, "rf": [0.0] * len(dates)})
        risk = compute_daily_risk_characteristics(daily, factors)
        self.assertIn("beta_252d", risk)
        self.assertAlmostEqual(float(risk["beta_252d"].dropna().iloc[-1]), 1.5, places=5)
        panel_keys = pd.DataFrame({"permno": [10], "date": [pd.Timestamp("2020-04-30")]})
        monthly = collapse_daily_risk_to_panel_months(risk, panel_keys)
        self.assertEqual(str(monthly["daily_risk_date"].iloc[0].date()), "2020-04-30")
        self.assertGreater(float(monthly["daily_obs_252d"].iloc[0]), 60)


if __name__ == "__main__":
    unittest.main()
