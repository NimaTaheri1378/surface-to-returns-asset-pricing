import unittest

import pandas as pd

from surface_returns.timing import is_accounting_known_by, next_month_return_window


class TimingTests(unittest.TestCase):
    def test_accounting_lag(self):
        self.assertFalse(is_accounting_known_by(pd.Timestamp("2022-12-31"), pd.Timestamp("2023-03-31")))
        self.assertTrue(is_accounting_known_by(pd.Timestamp("2022-12-31"), pd.Timestamp("2023-06-30")))

    def test_next_month_window(self):
        start, end = next_month_return_window(pd.Timestamp("2022-12-30"))
        self.assertEqual(str(start.date()), "2023-01-01")
        self.assertEqual(str(end.date()), "2023-01-31")


if __name__ == "__main__":
    unittest.main()
