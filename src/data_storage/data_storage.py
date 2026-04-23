# src/data_storage/data_storage.py

import pandas as pd
import logging

class DataStorage:
    """
    A class for storing data.
    """

    def __init__(self, file_path: str, format: str = 'csv'):
        """
        Initializes the DataStorage.

        Args:
            file_path (str): The path to the file to store the data in.
            format (str): The format to store the data in (csv or parquet).
        """
        self.file_path = file_path
        self.format = format
        self.logger = logging.getLogger(__name__)

    def save_data(self, data: list):
        """
        Saves the data to a file.

        Args:
            data (list): The data to save.
        """
        self.logger.info(f"Saving data to {self.file_path} in {self.format} format.")
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'number_of_trades', 
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        df = pd.DataFrame(data, columns=columns)
        
        try:
            if self.format == 'csv':
                df.to_csv(self.file_path, index=False)
            elif self.format == 'parquet':
                df.to_parquet(self.file_path, index=False)
            else:
                self.logger.error(f"Unsupported format: {self.format}")
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
