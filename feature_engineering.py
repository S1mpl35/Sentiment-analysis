"""
Module 3: Feature Engineering and Data Alignment
Merge stock data and sentiment data, construct feature matrices, and perform temporal split.
"""

import pandas as pd
import numpy as np
import os

def load_and_align(stock_path, sentiment_path):
    """
    Load stock and sentiment CSV files, align by date, and fill missing sentiment with 0.
    
    Parameters:
        stock_path (str): Path to stock data CSV (must contain 'Date', 'Return', 'Target_Direction')
        sentiment_path (str): Path to sentiment data CSV (must contain 'date' and sentiment columns)
    
    Returns:
        pd.DataFrame: Merged DataFrame with aligned dates and filled sentiment features.
    """
    print("=== Module 3: Feature Engineering & Alignment ===")
    
    # Check if files exist
    if not os.path.exists(stock_path):
        raise FileNotFoundError(f"Stock file not found: {stock_path}")
    if not os.path.exists(sentiment_path):
        raise FileNotFoundError(f"Sentiment file not found: {sentiment_path}")
    
    # Load data
    stock = pd.read_csv(stock_path)
    sentiment = pd.read_csv(sentiment_path)
    print(f"Loaded stock data: {len(stock)} rows, columns: {stock.columns.tolist()}")
    print(f"Loaded sentiment data: {len(sentiment)} rows, columns: {sentiment.columns.tolist()}")
    
    # Convert date columns to date objects (without time)
    stock['Date'] = pd.to_datetime(stock['Date']).dt.date
    sentiment['date'] = pd.to_datetime(sentiment['date']).dt.date
    
    # Check for duplicate dates in sentiment (should be unique after aggregation)
    if sentiment['date'].duplicated().any():
        print("Warning: Duplicate dates found in sentiment data. Keeping first occurrence.")
        sentiment = sentiment.drop_duplicates(subset=['date'], keep='first')
    
    # Merge: left join to keep all stock days, sentiment data aligned by date
    merged = pd.merge(stock, sentiment, left_on='Date', right_on='date', how='left')
    print(f"After merge: {len(merged)} rows")
    
    # Define sentiment columns to fill
    sent_cols = ['sentiment_mean', 'sentiment_std', 'sentiment_net', 'post_count', 'sentiment_diff']
    
    # Fill missing sentiment values with 0 (neutral)
    for col in sent_cols:
        if col in merged.columns:
            missing_count = merged[col].isna().sum()
            if missing_count > 0:
                print(f"Filling {missing_count} missing values in '{col}' with 0")
                merged[col] = merged[col].fillna(0)
        else:
            print(f"Warning: '{col}' not found in merged data. Creating column with zeros.")
            merged[col] = 0
    
    # Drop rows where essential target or return is missing
    required_cols = ['Return', 'Target_Direction']
    before_drop = len(merged)
    merged = merged.dropna(subset=required_cols).reset_index(drop=True)
    after_drop = len(merged)
    if before_drop != after_drop:
        print(f"Dropped {before_drop - after_drop} rows due to missing 'Return' or 'Target_Direction'")
    
    # Print date range of merged data
    if 'Date' in merged.columns:
        min_date = merged['Date'].min()
        max_date = merged['Date'].max()
        print(f"Merged data date range: {min_date} to {max_date} ({after_drop} trading days)")
    
    return merged

def build_features(df):
    """
    Build feature matrices for baseline (market only) and augmented (market + sentiment) models.
    
    Parameters:
        df (pd.DataFrame): Merged DataFrame containing market and sentiment features.
    
    Returns:
        X_market (np.ndarray): Market-only feature matrix.
        X_enhanced (np.ndarray): Market + sentiment feature matrix.
        y (np.ndarray): Target variable (direction).
        market_feature_names (list): Names of market features used.
        sentiment_feature_names (list): Names of sentiment features used.
    """
    # Define candidate market features (must exist in df)
    market_candidates = [
        'Return', 'Return_lag_1', 'Return_lag_2', 'Return_lag_3',
        'Volatility', 'Volatility_lag_1', 'Volatility_lag_2', 'Volatility_lag_3',
        'Volume_Change', 'Volume_Change_lag_1', 'Volume_Change_lag_2', 'Volume_Change_lag_3'
    ]
    
    # Define candidate sentiment features
    sentiment_candidates = [
        'sentiment_mean', 'sentiment_std', 'sentiment_net', 'post_count', 'sentiment_diff'
    ]
    
    # Keep only features that actually exist in the DataFrame
    market_features = [c for c in market_candidates if c in df.columns]
    sentiment_features = [c for c in sentiment_candidates if c in df.columns]
    
    # Check if any features are missing
    missing_market = set(market_candidates) - set(market_features)
    if missing_market:
        print(f"Note: Missing market features (will be excluded): {missing_market}")
    missing_sentiment = set(sentiment_candidates) - set(sentiment_features)
    if missing_sentiment:
        print(f"Note: Missing sentiment features (will be excluded): {missing_sentiment}")
    
    # Build feature matrices
    X_market = df[market_features].values
    X_enhanced = df[market_features + sentiment_features].values
    y = df['Target_Direction'].values
    
    # Print feature counts
    print(f"\nFeature summary:")
    print(f"  - Market features: {len(market_features)} -> {market_features}")
    print(f"  - Sentiment features: {len(sentiment_features)} -> {sentiment_features}")
    print(f"  - Total enhanced features: {len(market_features) + len(sentiment_features)}")
    print(f"  - Target distribution: Up={sum(y==1)}, Down={sum(y==0)}")
    
    return X_market, X_enhanced, y, market_features, sentiment_features

def time_split(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Split data chronologically into train, validation, and test sets.
    
    Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        train_ratio (float): Proportion for training (default 0.7).
        val_ratio (float): Proportion for validation (default 0.15).
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Ensure indices are within bounds
    if train_end == 0:
        raise ValueError("Training set too small. Increase train_ratio or dataset size.")
    if val_end <= train_end:
        raise ValueError("Validation set empty. Adjust train_ratio or val_ratio.")
    if val_end >= n:
        raise ValueError("Test set empty. Adjust ratios.")
    
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]
    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]
    
    print(f"\nData split (chronological):")
    print(f"  - Training set: {len(X_train)} samples ({train_ratio*100:.0f}%)")
    print(f"  - Validation set: {len(X_val)} samples ({val_ratio*100:.0f}%)")
    print(f"  - Test set: {len(X_test)} samples ({100-train_ratio*100-val_ratio*100:.0f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_split_data(output_path, X_train_m, X_val_m, X_test_m, X_train_e, X_val_e, X_test_e,
                    y_train, y_val, y_test, market_features, sentiment_features):
    """
    Save split data and feature names to a .npz file.
    """
    np.savez(output_path,
             X_train_m=X_train_m, X_val_m=X_val_m, X_test_m=X_test_m,
             X_train_e=X_train_e, X_val_e=X_val_e, X_test_e=X_test_e,
             y_train=y_train, y_val=y_val, y_test=y_test,
             feature_names_market=market_features,
             feature_names_enhanced=market_features + sentiment_features)
    print(f"\n✅ Split data saved to {output_path}")

if __name__ == "__main__":
    # Define file paths (adjust if needed)
    STOCK_FILE = "data/stock_AAPL.csv"
    SENTIMENT_FILE = "data/sentiment_daily.csv"
    MERGED_OUTPUT = "data/merged_dataset.csv"
    SPLIT_OUTPUT = "data/split_data.npz"
    
    # Step 1: Load and align data
    merged_df = load_and_align(STOCK_FILE, SENTIMENT_FILE)
    
    # Step 2: Save merged dataset for later use (e.g., backtesting)
    merged_df.to_csv(MERGED_OUTPUT, index=False)
    print(f"\n✅ Merged dataset saved to {MERGED_OUTPUT}")
    
    # Step 3: Build feature matrices
    X_market, X_enhanced, y, market_feats, sentiment_feats = build_features(merged_df)
    
    # Step 4: Chronological split
    X_train_m, X_val_m, X_test_m, y_train, y_val, y_test = time_split(X_market, y)
    X_train_e, X_val_e, X_test_e, _, _, _ = time_split(X_enhanced, y)
    
    # Step 5: Save split data
    save_split_data(SPLIT_OUTPUT,
                    X_train_m, X_val_m, X_test_m,
                    X_train_e, X_val_e, X_test_e,
                    y_train, y_val, y_test,
                    market_feats, sentiment_feats)
    
    print("\n=== Module 3 completed successfully ===")