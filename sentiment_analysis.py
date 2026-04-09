"""
Module 2: FinBERT sentiment analysis on Apple news data
Reads apple_news_data.csv, computes daily sentiment indicators, and saves sentiment_daily.csv
"""

import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

SENTIMENT_LABELS = ["negative", "neutral", "positive"]

def load_finbert():
    """
    Load pre-trained FinBERT model and tokenizer from Hugging Face.
    """
    print("Loading FinBERT model...")
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    print("✅ FinBERT model loaded")
    return tokenizer, model

def get_compound(text, tokenizer, model):
    """
    Compute compound sentiment score = P_pos - P_neg.
    
    Parameters:
        text: input news headline/title
        tokenizer: FinBERT tokenizer
        model: FinBERT model
    
    Returns:
        float: compound sentiment in range [-1, 1]
    """
    # Tokenize and truncate to max length
    inputs = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Get probabilities for [negative, neutral, positive]
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].numpy()
    # Compound = P_pos - P_neg
    return probs[2] - probs[0]

def main():
    print("=== Module 2: FinBERT Sentiment Analysis ===")
    
    # Path to input news data
    input_path = "data/apple_news_data.csv"
    output_path = "data/sentiment_daily.csv"
    
    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load news data
    news_df = pd.read_csv(input_path)
    print(f"Loaded raw news data: {len(news_df)} rows")
    print("News columns:", news_df.columns.tolist())
    
    # Identify text column (headline/title/news/text) and date column
    text_col = None
    date_col = None
    for col in news_df.columns:
        col_lower = col.lower()
        if 'headline' in col_lower or 'title' in col_lower or 'news' in col_lower or 'text' in col_lower:
            text_col = col
        if 'date' in col_lower or 'timestamp' in col_lower:
            date_col = col
    
    if text_col is None:
        raise KeyError(f"Cannot find text column. Available columns: {news_df.columns.tolist()}")
    if date_col is None:
        raise KeyError(f"Cannot find date column. Available columns: {news_df.columns.tolist()}")
    
    print(f"Using text column: '{text_col}', date column: '{date_col}'")
    
    # Convert date column to date objects (without time)
    news_df['date'] = pd.to_datetime(news_df[date_col], errors='coerce').dt.date
    # Drop rows with invalid dates
    before_drop = len(news_df)
    news_df = news_df.dropna(subset=['date'])
    after_drop = len(news_df)
    if before_drop != after_drop:
        print(f"Dropped {before_drop - after_drop} rows with invalid dates")
    
    # Filter by desired date range (must match Module 1)
    start_date = pd.to_datetime('2018-01-01').date()
    end_date = pd.to_datetime('2024-12-31').date()
    before_filter = len(news_df)
    news_df = news_df[(news_df['date'] >= start_date) & (news_df['date'] <= end_date)]
    after_filter = len(news_df)
    print(f"Filtered news by date range ({start_date} to {end_date}): {after_filter} rows (kept {after_filter} out of {before_filter})")
    
    if len(news_df) == 0:
        print("⚠️ No news in the selected date range.")
        print("   Consider adjusting start_date/end_date in the code or using a different dataset.")
        return
    
    # Load FinBERT model (may take ~1 minute on first run, downloads ~1.1GB)
    tokenizer, model = load_finbert()
    
    # Process each news item (using tqdm for progress bar)
    results = []
    # Optional: if dataset is huge, consider using batch processing, but here we keep per-item for simplicity
    for _, row in tqdm(news_df.iterrows(), total=len(news_df), desc="FinBERT analysis"):
        compound = get_compound(row[text_col], tokenizer, model)
        results.append({'date': row['date'], 'compound': compound})
    
    sentiment_df = pd.DataFrame(results)
    print(f"Generated sentiment scores for {len(sentiment_df)} news items")
    
    # Aggregate by date: compute mean, std, count, and net sentiment
    daily = sentiment_df.groupby('date').agg(
        sentiment_mean=('compound', 'mean'),
        sentiment_std=('compound', 'std'),
        post_count=('compound', 'count'),
        sentiment_net=('compound', lambda x: (x > 0).sum() - (x < 0).sum())
    ).reset_index()
    
    # Fill missing std (days with only one post) with 0
    daily['sentiment_std'] = daily['sentiment_std'].fillna(0)
    # Clip mean to [-1, 1] for safety
    daily['sentiment_mean'] = daily['sentiment_mean'].clip(-1, 1)
    # Signal-to-noise ratio (mean / std)
    daily['sentiment_diff'] = daily['sentiment_mean'] / (daily['sentiment_std'] + 1e-6)
    # Activity indicator (alias for post_count)
    daily['sentiment_activity'] = daily['post_count']
    
    # Save aggregated daily sentiment
    daily.to_csv(output_path, index=False)
    print(f"✅ Daily sentiment saved to {output_path} ({len(daily)} days)")
    
    # Print summary statistics
    print("\n--- Daily Sentiment Summary ---")
    print(f"Date range: {daily['date'].min()} to {daily['date'].max()}")
    print(f"Mean sentiment: {daily['sentiment_mean'].mean():.4f} (±{daily['sentiment_mean'].std():.4f})")
    print(f"Mean news count per day: {daily['post_count'].mean():.2f}")
    print(f"Days with news: {(daily['post_count'] > 0).sum()} out of {len(daily)}")

if __name__ == "__main__":
    main()