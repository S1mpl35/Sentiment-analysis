"""
Module 1: Load and preprocess AAPL stock data
Reads aapl_raw_data.csv, computes technical indicators, and saves stock_AAPL.csv
"""

import pandas as pd
import numpy as np
import os

def load_stock_data(input_path, output_path, start_date='2018-01-01', end_date='2024-12-31'):
    """
    Load raw AAPL stock CSV, compute technical indicators, and save processed data.

    Parameters:
        input_path (str): path to aapl_raw_data.csv
        output_path (str): where to save processed stock data
        start_date (str): start date for filtering (YYYY-MM-DD)
        end_date (str): end date for filtering (YYYY-MM-DD)

    Returns:
        pd.DataFrame: Processed stock data
    """
    print("=== Module 1: Load Stock Data ===")

    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load raw data
    df = pd.read_csv(input_path)
    print(f"Loaded raw data: {len(df)} rows")
    print("Original columns:", df.columns.tolist())

    # Standardize column names (case-insensitive mapping)
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'date' in col_lower:
            rename_map[col] = 'Date'
        elif col_lower in ['open', 'open_']:
            rename_map[col] = 'Open'
        elif col_lower in ['high', 'high_']:
            rename_map[col] = 'High'
        elif col_lower in ['low', 'low_']:
            rename_map[col] = 'Low'
        elif col_lower in ['close', 'close_', 'adj close']:
            rename_map[col] = 'Close'
        elif col_lower in ['volume', 'vol']:
            rename_map[col] = 'Volume'
    df = df.rename(columns=rename_map)
    print(f"Renamed columns: {df.columns.tolist()}")

    # Ensure required columns exist
    required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    # Convert date to date object and sort chronologically
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"Date range in raw data: {df['Date'].min()} to {df['Date'].max()}")

    # Filter by date range
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    before_filter = len(df)
    df = df[(df['Date'] >= start) & (df['Date'] <= end)]
    after_filter = len(df)
    print(f"Filtered stock data: {after_filter} rows ({start_date} to {end_date})")
    if after_filter == 0:
        raise ValueError(f"No data in the selected date range. Available range: {df['Date'].min()} to {df['Date'].max()}")

    # Compute simple returns
    df['Return'] = df['Close'].pct_change()
    # Compute log returns (for volatility calculation)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Volatility (20-day rolling standard deviation of log returns)
    df['Volatility'] = df['Log_Return'].rolling(20).std()

    # Simple moving averages
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()

    # Volume change rate
    df['Volume_Change'] = df['Volume'].pct_change()

    # Daily price range (normalized by close)
    df['Daily_Range'] = (df['High'] - df['Low']) / df['Close']

    # Lag features for returns, volatility, and volume change
    lags = [1, 2, 3, 5]
    for lag in lags:
        df[f'Return_lag_{lag}'] = df['Return'].shift(lag)
        df[f'Volatility_lag_{lag}'] = df['Volatility'].shift(lag)
        df[f'Volume_Change_lag_{lag}'] = df['Volume_Change'].shift(lag)

    # Target variables: next-day return and direction
    df['Target_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    df['Target_Direction'] = (df['Target_Return'] > 0).astype(int)

    # Drop rows with NaN (due to lag, rolling, or shift operations)
    before_drop = len(df)
    df = df.dropna().reset_index(drop=True)
    after_drop = len(df)
    print(f"Dropped {before_drop - after_drop} rows with missing values")

    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"✅ Stock data saved to {output_path} ({len(df)} rows)")
    print(f"   Final date range: {df['Date'].min()} to {df['Date'].max()}")
    return df

if __name__ == "__main__":
    # Adjust start_date and end_date as needed
    load_stock_data("data/aapl_raw_data.csv", "data/stock_AAPL.csv",
                    start_date='2018-01-01',
                    end_date='2024-12-31')