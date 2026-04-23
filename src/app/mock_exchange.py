
import pandas as pd

_MS_THRESHOLD = 10_000_000_000_000  # timestamps above this are in µs, not ms


class MockBinanceClient:
    def __init__(self, api_key, api_secret, testnet=True, mock_file=None):
        if not mock_file:
            raise ValueError("mock_file must be specified for MockBinanceClient")

        df = pd.read_csv(mock_file)

        # Normalize microsecond timestamps to milliseconds
        for col in ('timestamp', 'close_time'):
            if col in df.columns:
                mask = df[col] > _MS_THRESHOLD
                df.loc[mask, col] = df.loc[mask, col] // 1000

        self.mock_data = df
        self.current_index = 0
        self.data_length = len(self.mock_data)

    def get_historical_klines(self, symbol, interval, start_str=None, limit=500):
        if self.current_index >= self.data_length:
            return []

        end_index = min(self.current_index + limit, self.data_length)
        data_slice = self.mock_data.iloc[self.current_index:end_index]
        self.current_index += 1

        # Mirror the real Binance API: timestamps are ints, all other numeric
        # fields are strings, number_of_trades is int, ignore is string "0".
        ohlcv_list = []
        for _, row in data_slice.iterrows():
            ohlcv_list.append([
                int(row['timestamp']),
                str(row['open']),
                str(row['high']),
                str(row['low']),
                str(row['close']),
                str(row['volume']),
                int(row.get('close_time', 0)),
                str(row.get('Volume USDT', '0')),
                int(row.get('tradeCount', 0)),
                str(row.get('taker_buy_base_asset_volume', '0')),
                str(row.get('taker_buy_quote_asset_volume', '0')),
                str(row.get('ignore', '0')),
            ])

        return ohlcv_list

    def create_order(self, symbol, type, side, amount):
        print(f"Mock Order: {side} {amount} of {symbol} at market price.")
        idx = min(self.current_index - 1, self.data_length - 1)
        current_price = float(self.mock_data['close'].iloc[idx])
        return {
            'info': {'orderId': 'mock_order_123'},
            'id': 'mock_order_123',
            'clientOrderId': 'mock_client_order_123',
            'timestamp': pd.Timestamp.now().timestamp() * 1000,
            'datetime': str(pd.Timestamp.now()),
            'status': 'closed',
            'symbol': symbol,
            'type': type,
            'side': side,
            'price': current_price,
            'amount': amount,
            'filled': amount,
            'remaining': 0,
            'cost': amount * current_price,
            'trades': [],
            'fee': {},
        }

    def fetch_balance(self):
        return {
            'free': {'USDT': 10000.0, 'BTC': 0.0},
            'total': {'USDT': 10000.0, 'BTC': 0.0},
        }
