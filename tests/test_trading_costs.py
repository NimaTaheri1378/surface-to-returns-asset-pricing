import unittest

import pandas as pd

from surface_returns.trading_costs import (
    aggregate_taq_daily_costs,
    calibrated_trade_cost,
    fill_taq_costs_with_calibration,
    merge_taq_costs_to_panel,
    portfolio_taq_costs,
)


class TradingCostTests(unittest.TestCase):
    def test_taq_quote_and_trade_aggregation(self):
        quotes = pd.DataFrame(
            {
                "sym_root": ["abc", "ABC", "XYZ"],
                "bid": [9.90, 9.95, 20.0],
                "ofr": [10.10, 10.05, 20.2],
                "bidsiz": [100, 200, 300],
                "ofrsiz": [100, 200, 300],
            }
        )
        trades = pd.DataFrame(
            {
                "sym_root": ["ABC", "ABC", "XYZ"],
                "price": [10.0, 10.1, 20.1],
                "size": [100, 150, 200],
            }
        )
        costs = aggregate_taq_daily_costs(quotes, trades, trade_date="2020-01-31")
        abc = costs[costs["ticker"].eq("ABC")].iloc[0]
        self.assertAlmostEqual(float(abc["taq_half_spread"]), 0.0075, places=6)
        self.assertEqual(int(abc["taq_trade_obs"]), 2)
        self.assertGreater(float(abc["taq_dollar_volume"]), 0)

    def test_merge_taq_costs_to_panel_uses_permno_date_ticker_map(self):
        panel = pd.DataFrame({"permno": [1], "date": [pd.Timestamp("2020-01-31")]})
        mapping = pd.DataFrame({"permno": [1], "date": [pd.Timestamp("2020-01-31")], "ticker": ["abc"]})
        costs = pd.DataFrame(
            {
                "ticker": ["ABC"],
                "date": [pd.Timestamp("2020-01-31")],
                "taq_half_spread": [0.001],
            }
        )
        merged = merge_taq_costs_to_panel(panel, costs, mapping)
        self.assertAlmostEqual(float(merged["taq_half_spread"].iloc[0]), 0.001)

    def test_portfolio_taq_costs_adds_impact_to_spread(self):
        trades = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-01-31")],
                "permno": [1],
                "abs_trade_weight": [0.1],
            }
        )
        costs = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-01-31")],
                "permno": [1],
                "taq_half_spread": [0.001],
                "taq_intraday_vol": [0.02],
                "taq_dollar_volume": [10.0],
            }
        )
        direct = calibrated_trade_cost(
            costs["taq_half_spread"],
            costs["taq_intraday_vol"],
            trades["abs_trade_weight"],
            costs["taq_dollar_volume"],
            portfolio_value=1.0,
        )
        self.assertGreater(float(direct.iloc[0]), 0.001)
        monthly = portfolio_taq_costs(trades, costs, portfolio_value=1.0)
        self.assertGreater(float(monthly["taq_total_cost"].iloc[0]), 0.0)

    def test_fill_taq_costs_calibrates_missing_rows(self):
        panel = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28")],
                "permno": [1, 2],
                "market_equity": [1_000_000_000.0, 2_000_000_000.0],
                "dollar_volume": [210_000_000.0, 420_000_000.0],
                "idio_vol_252d": [0.20, 0.30],
                "taq_half_spread": [0.0015, None],
                "taq_intraday_vol": [0.012, None],
                "taq_dollar_volume": [10_000_000.0, None],
            }
        )
        filled = fill_taq_costs_with_calibration(panel)
        self.assertEqual(filled["taq_cost_source"].tolist(), ["exact_taq", "calibrated_taq"])
        self.assertTrue(filled[["taq_half_spread", "taq_intraday_vol", "taq_dollar_volume"]].notna().all().all())
        self.assertGreater(float(filled.loc[1, "taq_half_spread"]), 0.0)


if __name__ == "__main__":
    unittest.main()
