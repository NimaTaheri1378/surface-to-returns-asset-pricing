import unittest

import pandas as pd

from surface_returns.backtest import (
    long_short_decile_weights,
    long_short_portfolio_returns,
    one_way_transaction_cost,
    performance_summary,
    portfolio_turnover,
)


class BacktestTests(unittest.TestCase):
    def test_transaction_cost_nonnegative(self):
        cost = one_way_transaction_cost(
            pd.Series([0.001]),
            pd.Series([0.20]),
            pd.Series([0.01]),
            eta=0.10,
        )
        self.assertGreater(float(cost.iloc[0]), 0.001)

    def test_turnover(self):
        weights = pd.DataFrame(
            {
                "date": ["2022-12-31", "2022-12-31", "2023-01-31", "2023-01-31"],
                "permno": [1, 2, 1, 2],
                "weight": [0.5, -0.5, 0.2, -0.2],
            }
        )
        turnover = portfolio_turnover(weights)
        self.assertAlmostEqual(float(turnover.iloc[0]), 1.0)
        self.assertAlmostEqual(float(turnover.iloc[1]), 0.6)

    def test_decile_weights_are_dollar_neutral(self):
        frame = pd.DataFrame(
            {
                "date": ["2023-01-31"] * 20,
                "permno": list(range(20)),
                "pred": list(range(20)),
                "next_ret": [0.01] * 20,
            }
        )
        weights = long_short_decile_weights(frame, min_assets=20)
        self.assertEqual(len(weights), 4)
        self.assertAlmostEqual(float(weights["weight"].sum()), 0.0)
        self.assertAlmostEqual(float(weights["weight"].abs().sum()), 1.0)
        self.assertEqual(set(weights["side"]), {"long", "short"})

    def test_long_short_returns_apply_costs(self):
        weights = pd.DataFrame(
            {
                "date": ["2023-01-31", "2023-01-31"],
                "permno": [1, 2],
                "side": ["long", "short"],
                "weight": [0.5, -0.5],
            }
        )
        returns = pd.DataFrame(
            {
                "date": ["2023-01-31", "2023-01-31"],
                "permno": [1, 2],
                "next_ret": [0.10, -0.02],
                "half_spread": [0.001, 0.001],
            }
        )
        monthly = long_short_portfolio_returns(weights, returns, half_spread_col="half_spread")
        self.assertAlmostEqual(float(monthly.loc[0, "gross_return"]), 0.06)
        self.assertLess(float(monthly.loc[0, "net_return"]), 0.06)
        self.assertGreater(float(monthly.loc[0, "total_cost"]), 0.0)

    def test_performance_summary_reports_months(self):
        monthly = pd.DataFrame({"ret": [0.01, -0.02, 0.03]})
        summary = performance_summary(monthly, "ret")
        self.assertEqual(summary["months"], 3)
        self.assertIn("max_drawdown", summary)


if __name__ == "__main__":
    unittest.main()
