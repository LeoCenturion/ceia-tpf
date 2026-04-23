# tests/data_preprocessing/test_splitting.py

import unittest
import pandas as pd
from src.data_preprocessing.splitting import split_data

class TestSplitting(unittest.TestCase):
    """
    Tests for the splitting functions.
    """

    def test_split_data(self):
        """
        Test splitting the data.
        """
        data = {'col1': range(100)}
        df = pd.DataFrame(data)
        
        train_df, val_df, test_df = split_data(df)
        
        self.assertEqual(len(train_df), 70)
        self.assertEqual(len(val_df), 15)
        self.assertEqual(len(test_df), 15)
        self.assertTrue(train_df.index.max() < val_df.index.min())
        self.assertTrue(val_df.index.max() < test_df.index.min())

if __name__ == '__main__':
    unittest.main()
