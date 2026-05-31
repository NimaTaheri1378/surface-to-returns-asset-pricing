import unittest

import pandas as pd

from surface_returns.linkage import validate_nonoverlapping_links


class LinkageTests(unittest.TestCase):
    def test_overlap_detection(self):
        links = pd.DataFrame(
            {
                "permno": [1, 1, 2],
                "start": ["2020-01-01", "2020-06-01", "2020-01-01"],
                "end": ["2020-12-31", "2021-01-01", "2020-03-01"],
            }
        )
        overlaps = validate_nonoverlapping_links(links, "permno", "start", "end")
        self.assertEqual(len(overlaps), 1)


if __name__ == "__main__":
    unittest.main()
