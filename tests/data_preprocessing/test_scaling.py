# tests/data_preprocessing/test_scaling.py

import unittest

import numpy as np
import pandas as pd

from src.app.data_preprocessing.scaling import scale_data


class TestScaling(unittest.TestCase):
    """
    Tests for the scaling functions.
    """

    def test_scale_data(self):
        """
        Test scaling the data.
        """
        data = {"col1": [1, 2, 3, 4, 5], "col2": [10, 20, 30, 40, 50]}
        df = pd.DataFrame(data)

        scaled_df = scale_data(df)

        self.assertIsInstance(scaled_df, pd.DataFrame)
        self.assertEqual(len(scaled_df), 5)
        self.assertTrue(np.allclose(np.std(scaled_df, ddof=0), 1))


if __name__ == "__main__":
    unittest.main()
