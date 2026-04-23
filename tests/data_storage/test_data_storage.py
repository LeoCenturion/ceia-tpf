# tests/data_storage/test_data_storage.py

import os
import unittest

import pandas as pd

from src.data_storage.data_storage import DataStorage


class TestDataStorage(unittest.TestCase):
    """
    Tests for the DataStorage class.
    """

    def setUp(self):
        """
        Set up the test environment.
        """
        self.csv_file = 'test_data.csv'
        self.parquet_file = 'test_data.parquet'
        self.data = [
            [1503388800000, '4235.43000000', '4235.43000000', '4235.43000000', '4235.43000000', '0.00000000', 1503388859999, '0.00000000', 0, '0.00000000', '0.00000000', '0']
        ]

    def tearDown(self):
        """
        Clean up the test environment.
        """
        if os.path.exists(self.csv_file):
            os.remove(self.csv_file)
        if os.path.exists(self.parquet_file):
            os.remove(self.parquet_file)

    def test_save_data_csv(self):
        """
        Test saving data to a CSV file.
        """
        storage = DataStorage(self.csv_file, format='csv')
        storage.save_data(self.data)
        
        self.assertTrue(os.path.exists(self.csv_file))
        
        df = pd.read_csv(self.csv_file)
        self.assertEqual(len(df), 1)

    def test_save_data_parquet(self):
        """
        Test saving data to a Parquet file.
        """
        storage = DataStorage(self.parquet_file, format='parquet')
        storage.save_data(self.data)
        
        self.assertTrue(os.path.exists(self.parquet_file))
        
        df = pd.read_parquet(self.parquet_file)
        self.assertEqual(len(df), 1)

    def test_unsupported_format(self):
        """
        Test that an unsupported format is handled correctly.
        """
        storage = DataStorage('test.txt', format='txt')
        storage.save_data(self.data)
        self.assertFalse(os.path.exists('test.txt'))

if __name__ == '__main__':
    unittest.main()
