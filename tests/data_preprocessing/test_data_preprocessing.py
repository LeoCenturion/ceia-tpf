# tests/data_preprocessing/test_data_preprocessing.py

import unittest

import pandas as pd

from src.app.data_preprocessing.data_preprocessing import (
    clean_data,
    load_data,
    prepare_data,
)


class TestDataPreprocessing(unittest.TestCase):
    """
    Tests for the DataPreprocessing class.
    """

    def test_load_data(self):
        """
        Test loading data from a CSV file.
        """
        # Create a dummy CSV file for testing
        data = {"col1": [1, 2], "col2": [3, 4]}
        df = pd.DataFrame(data)
        df.to_csv("test.csv", index=False)

        loaded_df = load_data("test.csv")
        self.assertIsInstance(loaded_df, pd.DataFrame)
        self.assertEqual(len(loaded_df), 2)

    def test_load_data_file_not_found(self):
        """
        Test that an empty DataFrame is returned when the file is not found.
        """
        loaded_df = load_data("non_existent_file.csv")
        self.assertIsInstance(loaded_df, pd.DataFrame)
        self.assertEqual(len(loaded_df), 0)

    def test_prepare_data(self):
        """
        Test the prepare_data function.
        """
        data = {"col1": [1, 2], "col2": [3, 4]}
        df = pd.DataFrame(data)

        prepared_df = prepare_data(df)
        self.assertIsInstance(prepared_df, pd.DataFrame)
        self.assertEqual(len(prepared_df), 2)

    def test_clean_data(self):
        """
        Test the clean_data function.
        """
        data = {"col1": [1, 2, 3, 4, 100], "col2": [5, 6, 7, 8, -100]}
        df = pd.DataFrame(data)

        cleaned_df = clean_data(df)
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertEqual(len(cleaned_df), 5)
        self.assertTrue(cleaned_df["col1"].max() < 100)
        self.assertTrue(cleaned_df["col2"].min() > -100)
