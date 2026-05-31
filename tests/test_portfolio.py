import unittest

import pandas as pd

from surface_returns.portfolio import buffered_memberships, sector_balanced_weights, sic_to_ff12


class PortfolioTests(unittest.TestCase):
    def test_sic_to_sector(self):
        self.assertEqual(sic_to_ff12(6020), "Finance")
        self.assertEqual(sic_to_ff12(1311), "Energy")
        self.assertEqual(sic_to_ff12(None), "Other")

    def test_buffered_memberships_keeps_names_inside_exit_band(self):
        frame = pd.DataFrame(
            {
                "date": ["2020-01-31"] * 100 + ["2020-02-29"] * 100,
                "permno": list(range(100)) * 2,
                "pred": list(range(100)) + list(range(5, 100)) + [0, 1, 2, 3, 4],
            }
        )
        members = buffered_memberships(frame, entry_pct=0.1, exit_pct=0.2, min_assets=50)
        self.assertGreater(len(members), 20)
        self.assertIn("side", members)

    def test_sector_balanced_weights_are_dollar_neutral(self):
        members = pd.DataFrame(
            {
                "date": ["2020-01-31"] * 4,
                "permno": [1, 2, 3, 4],
                "side": ["long", "long", "short", "short"],
                "sector": ["A", "B", "A", "B"],
                "pred": [0.9, 0.8, 0.1, 0.2],
            }
        )
        weights = sector_balanced_weights(members)
        self.assertAlmostEqual(float(weights["weight"].sum()), 0.0)
        self.assertAlmostEqual(float(weights["weight"].abs().sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
