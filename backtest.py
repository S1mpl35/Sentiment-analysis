"""
Module 5: Simulated Trading Backtest with automatic signal direction selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier

def evaluate_strategy(predictions, prices, dates, initial_capital=10000, threshold=0.5, transaction_cost=0.001):
    """
    Evaluate a trading strategy given predictions (probabilities of up movement).
    
    Parameters:
        predictions (array): predicted probabilities of upward movement (0 to 1)
        prices (array): daily close prices
        dates (array): corresponding dates
        initial_capital (float): starting capital
        threshold (float): probability threshold for taking a long position
        transaction_cost (float): one-way transaction cost (e.g., 0.001 = 0.1%)
    
    Returns:
        equity (array): equity curve over the period
        total_return (float): cumulative return
        sharpe (float): annualised Sharpe ratio
        max_dd (float): maximum drawdown
        win_rate (float): proportion of days with positive returns
    """
    # Generate trading signals (1 = long, 0 = flat)
    signals = (predictions > threshold).astype(int)
    # Lag signals by one day (decision at close, execute at next open)
    signals_lagged = np.roll(signals, 1)
    signals_lagged[0] = 0

    # Daily market returns (next day's return relative to today's close)
    daily_returns = prices[1:] / prices[:-1] - 1
    daily_returns = np.insert(daily_returns, 0, 0)

    # Strategy returns: position size * market return
    strategy_returns = signals_lagged * daily_returns
    # Transaction cost: incurred when position changes (buy or sell)
    trade_penalty = np.abs(np.diff(signals_lagged.astype(int))) * transaction_cost
    trade_penalty = np.insert(trade_penalty, 0, 0)
    strategy_returns = strategy_returns - trade_penalty

    # Equity curve
    equity = initial_capital * (1 + strategy_returns).cumprod()
    total_return = (equity[-1] - initial_capital) / initial_capital
    # Annualised Sharpe ratio (assuming 252 trading days, zero risk-free rate)
    sharpe = (strategy_returns.mean() / (strategy_returns.std() + 1e-6)) * np.sqrt(252)
    # Maximum drawdown
    running_max = np.maximum.accumulate(equity)
    max_dd = (equity / running_max - 1).min()
    # Win rate: proportion of days with positive strategy returns
    win_rate = (strategy_returns > 0).mean()
    
    return equity, total_return, sharpe, max_dd, win_rate

def plot_equity_curve(equity_curve, dates, benchmark_equity, save_path="backtest_curve.png"):
    """
    Plot equity curve: Blue = Buy & Hold, Gray = Strategy.
    
    Parameters:
        equity_curve (array): strategy equity values
        dates (array): dates corresponding to equity curve
        benchmark_equity (array): buy-and-hold equity values
        save_path (str): where to save the figure
    """
    plt.figure(figsize=(12, 6))
    # Blue line = buy & hold benchmark
    plt.plot(dates, benchmark_equity[:len(dates)], label='Buy & Hold', linewidth=2, color='blue')
    # Gray line = strategy equity
    plt.plot(dates, equity_curve, label='Strategy Equity', linewidth=1.5, linestyle='--', color='gray')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Simulated Trading Backtest: Buy & Hold (Blue) vs Strategy (Gray)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✅ Equity curve saved to {save_path}")

def main():
    print("=== Module 5: Simulated Trading Backtest with Signal Direction Optimization ===")
    
    # Check if merged dataset exists
    data_path = "data/merged_dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Merged dataset not found: {data_path}. Please run Module 3 first.")
    
    # Load merged dataset
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"Loaded merged dataset: {len(df)} rows, date range {df['Date'].min()} to {df['Date'].max()}")
    
    # Features (same as in Module 3)
    market_features = ['Return', 'Volatility', 'Volume_Change', 'Return_lag_1', 'Return_lag_2']
    sentiment_features = ['sentiment_mean', 'sentiment_std', 'sentiment_net']
    available = [c for c in market_features + sentiment_features if c in df.columns]
    missing = set(market_features + sentiment_features) - set(available)
    if missing:
        print(f"Warning: Missing features: {missing}. They will be excluded.")
    X = df[available].fillna(0).values
    y = df['Target_Direction'].values
    prices = df['Close'].values
    dates = df['Date'].values
    print(f"Feature matrix shape: {X.shape}, target shape: {y.shape}")
    
    # Split into train, validation, test (60% train, 20% val, 20% test)
    n = len(X)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    prices_val = prices[train_end:val_end]
    prices_test = prices[val_end:]
    dates_test = dates[val_end:]
    print(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # Train Random Forest on training set
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("✅ Model trained")
    
    # Get probabilities on validation set
    probs_val = model.predict_proba(X_val)[:, 1]
    
    # Evaluate both original and reversed signals on validation set
    print("\nEvaluating signal directions on validation set...")
    equity_orig, ret_orig, sharpe_orig, dd_orig, wr_orig = evaluate_strategy(
        probs_val, prices_val, dates[train_end:val_end]
    )
    equity_rev, ret_rev, sharpe_rev, dd_rev, wr_rev = evaluate_strategy(
        1 - probs_val, prices_val, dates[train_end:val_end]
    )
    
    print("\n=== Validation Performance ===")
    print(f"Original signal direction: cumulative return = {ret_orig*100:.2f}%, Sharpe = {sharpe_orig:.3f}")
    print(f"Reversed signal direction: cumulative return = {ret_rev*100:.2f}%, Sharpe = {sharpe_rev:.3f}")
    
    # Choose the better direction based on cumulative return
    use_reversed = ret_rev > ret_orig
    if use_reversed:
        print("\n✅ Using REVERSED signal direction (buy when predicted probability < 0.5)")
    else:
        print("\n✅ Using ORIGINAL signal direction (buy when predicted probability > 0.5)")
    
    # Apply chosen direction on test set
    probs_test = model.predict_proba(X_test)[:, 1]
    if use_reversed:
        final_probs = 1 - probs_test
    else:
        final_probs = probs_test
    
    # Backtest on test set
    print("\nRunning backtest on test set...")
    equity_test, total_return, sharpe, max_dd, win_rate = evaluate_strategy(
        final_probs, prices_test, dates_test
    )
    
    # Buy & hold benchmark on test set
    benchmark_equity = 10000 * (prices_test / prices_test[0])
    # Align lengths (in case of slight mismatch)
    min_len = min(len(equity_test), len(benchmark_equity))
    equity_test = equity_test[:min_len]
    benchmark_equity = benchmark_equity[:min_len]
    dates_test_aligned = dates_test[:min_len]
    
    # Print results
    print("\n=== Test Set Results ===")
    print(f"Initial capital:      $10,000")
    print(f"Final capital:        ${equity_test[-1]:,.2f}")
    print(f"Cumulative return:    {total_return*100:.2f}%")
    print(f"Annualised Sharpe:    {sharpe:.3f}")
    print(f"Maximum drawdown:     {max_dd*100:.2f}%")
    print(f"Win rate:             {win_rate*100:.2f}%")
    
    # Plot equity curve
    plot_equity_curve(equity_test, dates_test_aligned, benchmark_equity, save_path="backtest_curve.png")
    print("✅ Module 5 completed")

if __name__ == "__main__":
    main()