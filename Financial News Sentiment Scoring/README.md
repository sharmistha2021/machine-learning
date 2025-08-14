# Financial News Sentiment Analysis with FinBERT

## Overview
This project performs **sentiment analysis on financial news headlines** using **FinBERT**, a BERT-based model fine-tuned for financial text. The goal is to extract **daily sentiment scores** that can be used as features for stock trend prediction or other financial analyses.

The workflow includes:
1. Loading filtered financial news.
2. Applying FinBERT for sentiment classification.
3. Adjusting sentiment results using keyword-based rules.
4. Mapping sentiment labels to numeric scores.
5. Aggregating scores by day.
6. Filling missing dates to create a continuous daily time series.

---

## Detailed Workflow

### 1. Load filtered news data
- The dataset `filtered_news_df.csv` contains financial news headlines along with their posting date and stock code.
- We load this CSV into a Pandas DataFrame for processing.

### 2. Apply FinBERT sentiment analysis
- FinBERT is pre-trained on financial text and classifies each news item as:
  - **Positive**
  - **Neutral**
  - **Negative**
- We use the HuggingFace `transformers` pipeline to apply FinBERT to each news headline.
- Error handling is included to skip invalid or empty text.

### 3. Rule-based adjustments for Neutral labels
- Sometimes FinBERT predicts “Neutral” even if the news implies positive or negative sentiment.
- We apply keyword rules to adjust Neutral cases:
  - Words like `"profit"`, `"increase"`, `"growth"` → reclassify as **Positive**.
  - Words like `"loss"`, `"decrease"`, `"decline"` → reclassify as **Negative**.

### 4. Map sentiment labels to numeric scores
- Convert textual labels into numerical values for easier integration with models:
  - Positive → `1`
  - Neutral → `0`
  - Negative → `-1`

### 5. Aggregate daily sentiment
- Multiple news articles may exist for the same day.
- We calculate the **average sentiment score per day**, creating a daily sentiment index.
- This is saved as `daily_sentiment_scores.csv`.

### 6. Fill missing dates
- Financial markets have continuous dates, but some days may have no news.
- We generate a **full date range** and reindex the sentiment data to cover every day.
- Missing sentiment scores are forward-filled from the last available value.
- This ensures a **continuous time series**, ready to merge with stock prices.

---

## Output File
`filled_sentiment_series.csv` – Daily sentiment series with missing dates filled.

---

## Applications
- Merge with historical stock prices for **predictive modeling** (GRU, LSTM, BiLSTM).  
- Analyze how news sentiment affects stock price movements.  
- Create **event-driven trading strategies** based on daily sentiment.

---

## Dependencies
- Python >= 3.8  
- pandas  
- transformers  
- torch  

Install with:
```bash
pip install pandas transformers torch
