import unittest

import numpy as np
import pandas as pd

from surface_returns.conditional_sdf import (
    classify_conditional_sdf_features,
    make_walk_forward_sdf_splits,
    month_arrays,
    prepare_conditional_sdf_frame,
    sdf_pricing_error_summary,
    select_conditional_sdf_features,
)


class ConditionalSDFTests(unittest.TestCase):
    def test_feature_selection_and_month_arrays(self):
        dates = pd.date_range("2020-01-31", periods=4, freq="ME")
        frame = pd.DataFrame(
            {
                "date": np.repeat(dates, 3),
                "permno": list(range(12)),
                "next_ret": np.linspace(-0.02, 0.03, 12),
                "mean_iv": np.linspace(0.1, 0.5, 12),
                "surface_ae_01": np.linspace(-1.0, 1.0, 12),
                "mostly_missing": [np.nan] * 11 + [1.0],
            }
        )
        features = select_conditional_sdf_features(frame, min_nonmissing=0.8)
        self.assertIn("mean_iv", features)
        self.assertIn("surface_ae_01", features)
        self.assertNotIn("mostly_missing", features)
        prepared = prepare_conditional_sdf_frame(frame, features, min_assets_per_month=3)
        self.assertEqual(len(prepared), 12)
        self.assertAlmostEqual(float(prepared.groupby("date")["mean_iv"].mean().abs().max()), 0.0, places=6)
        arrays = month_arrays(prepared, features)
        self.assertEqual(len(arrays), 4)
        self.assertEqual(arrays[0][1].shape[1], len(features))

    def test_state_features_are_not_cross_sectionally_zeroed(self):
        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        frame = pd.DataFrame(
            {
                "date": np.repeat(dates, 4),
                "permno": list(range(12)),
                "next_ret": np.linspace(-0.02, 0.03, 12),
                "mean_iv": np.linspace(0.1, 0.5, 12),
                "vix_eom": np.repeat([12.0, 18.0, 24.0], 4),
                "fred_unrate": np.repeat([3.5, 3.7, 4.0], 4),
            }
        )
        features = ["mean_iv", "vix_eom", "fred_unrate"]
        groups = classify_conditional_sdf_features(features)

        prepared = prepare_conditional_sdf_frame(frame, groups.ordered, min_assets_per_month=4, state_cols=groups.state)

        self.assertEqual(groups.surface, ["mean_iv"])
        self.assertEqual(groups.state, ["vix_eom", "fred_unrate"])
        self.assertAlmostEqual(float(prepared.groupby("date")["mean_iv"].mean().abs().max()), 0.0, places=6)
        self.assertEqual(prepared.groupby("date")["vix_eom"].first().tolist(), [12.0, 18.0, 24.0])
        self.assertGreater(float(prepared["vix_eom"].std()), 0.0)

    def test_walk_forward_splits_and_pricing_summary(self):
        dates = list(pd.date_range("2020-01-31", periods=8, freq="ME"))
        splits = make_walk_forward_sdf_splits(dates, min_train_months=4, test_months=2)
        self.assertEqual(len(splits), 2)
        self.assertEqual(splits[0].fold, 1)
        self.assertEqual(len(splits[0].train_dates), 4)
        summary = sdf_pricing_error_summary(pd.Series([0.01, -0.02, 0.03]))
        self.assertEqual(summary["n"], 3)
        self.assertGreater(summary["rms"], summary["mean_abs"] * 0.8)


if __name__ == "__main__":
    unittest.main()
