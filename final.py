#final Project by Lincoln Ledet
#This program will analyze the sentiment of news articles related to a list of stocks and calculate the correlation between the sentiment and the daily stock price change. 
# The program will also display a bar chart showing the daily stock price change and the average sentiment score for each stock.

import feedparser
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# List of tickers
tickers = [
    'AMZN', 'MSFT', 'NFLX', 'BAC', 'XOM', 'PG', 'AVAV', 'UNH',
    'GOOGL', 'META', 'DIS', 'V', 'MA', 'KO', 'PEP', 'PFE', 
]

# Mapping from ticker to company name
# Extended mapping from ticker to company name
ticker_company_map = {
    'AMZN': 'Amazon',
    'MSFT': 'Microsoft',
    'NFLX': 'Netflix',
    'JPM': 'JPMorgan Chase',
    'AAPL': 'Apple',
    'TSLA': 'Tesla',
    'NVDA': 'NVIDIA',
    'HD': 'Home Depot',
    'BAC': 'Bank of America',
    'XOM': 'ExxonMobil',
    'PG': 'Procter & Gamble',
    'AVAV': 'AeroVironment',
    'UNH': 'UnitedHealth',
    'GOOGL': 'Alphabet (Google)',
    'META': 'Meta Platforms (Facebook)',
    'DIS': 'Disney',
    'V': 'Visa',
    'MA': 'Mastercard',
    'KO': 'Coca-Cola',
    'PEP': 'PepsiCo',
    'PFE': 'Pfizer',
    'MRK': 'Merck & Co.',
    'T': 'AT&T',
    'VZ': 'Verizon Communications',
    'CVX': 'Chevron',
    'WMT': 'Walmart',
    'COST': 'Costco',
    'CSCO': 'Cisco Systems',
    'ORCL': 'Oracle',
    'IBM': 'IBM',
    'NKE': 'Nike',
    'MCD': 'McDonald’s',
    'INTC': 'Intel',
    'GE': 'General Electric',
    'BA': 'Boeing',
    'CAT': 'Caterpillar',
    'MMM': '3M',
    'GM': 'General Motors',
    'F': 'Ford Motor',
    'UPS': 'United Parcel Service',
    'FDX': 'FedEx',
    'LOW': 'Lowe’s',
    'ADBE': 'Adobe',
    'CRM': 'Salesforce',
    'PYPL': 'PayPal',
    'SQ': 'Block (Square)',
    'ZM': 'Zoom Video Communications',
    'ROKU': 'Roku',
    'TWTR': 'Twitter',
    'SNAP': 'Snapchat (Snap Inc.)'
}


def GetTickerData(ticker, keyword):
    yahoo_url = 'https://finance.yahoo.com/rss/headline?s='
    data = []
    feed = feedparser.parse(yahoo_url + ticker)
    for i, entry in enumerate(feed.entries):
        if keyword.lower() in entry.summary.lower(): #makes sure company name is in the title
            data.append({
                'summary': entry.summary,
            })
            if i == 30:
                break
    return data

# Load the transformer sentiment model
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def AnalyzeSentiment_TB(summaries):
    sentiments = []
    scores = []
    for summary in summaries:
        inputs = tokenizer(summary, return_tensors="pt", truncation=True, padding=True) # tokenizes the summary
        outputs = model(**inputs) # passes the tokenized summary to the model
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1) # computes the probabilities of the sentiment
        sentiment = torch.argmax(probabilities).item() # gets the sentiment with the highest probability
        sentiment_score = probabilities[0][sentiment].item() # gets the score of the sentiment
        sentiment_label = ['negative', 'neutral', 'positive'][sentiment] # assigns the sentiment label

        sentiments.append((sentiment_label, sentiment_score))
        # Convert sentiment to numeric score similar to original code:
        # positive -> +score
        # negative -> -score
        # neutral -> -0.4 * score
        if sentiment_label == 'positive':
            final_score = sentiment_score
        elif sentiment_label == 'negative':
            final_score = -sentiment_score
        else:
            final_score = -0.4 * sentiment_score

        scores.append(final_score)

    print("Printing sentiment and score of various articles...")
    if summaries:
        print("Articles summary: ", summaries[0])
        print("Sentiment: ", sentiments[0])
    return sentiments, np.mean(scores) if scores else 0

def GetDailyChange(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if hist.empty:
            raise ValueError("No data available for the stock.")
        close = hist['Close'].iloc[-1]
        open_ = hist['Open'].iloc[-1]
        percent_change = ((close - open_) / open_) * 100
        return round(percent_change, 2)
    except Exception as e:
        return f"Error: {e}"

# Function to get Custom Model Sentiment score using Decision Tree
def AnalyzeSentiment_Custom(summaries):
    # Vectorize the incoming summaries using the previously fitted vectorizer
    if len(summaries) == 0:
        return [], 0
    print ("Printing sentiment and score of various articles...")
    print ("Articles summary: ", summaries)

    # Convert the predicted labels to sentiment scores
    # Assuming 0=negative, 1=neutral, 2=positive
    # We'll use a similar scoring scheme as the transformer:
    # positive: +1
    # negative: -1
    # neutral: let's assign -0.4
    # NOTE: Since this model doesn't give probabilities easily, we simplify:
    # You could also consider predict_proba if needed.
    sentiments = []
    scores = []
        
    return sentiments, np.mean(scores) if scores else 0

#RUN BOTH MODELS ON RSS SUMMARIES AND COMPARE

stock_data_list = []

for ticker in tickers:
    company_name = ticker_company_map.get(ticker, '')
    if company_name:
        ticker_data = GetTickerData(ticker, company_name)
        summaries = [entry['summary'] for entry in ticker_data]

        # Transformer-based sentiment analysis
        tb_sentiments, tb_avg_score = AnalyzeSentiment_TB(summaries)

        # Custom Decision Tree sentiment analysis
        custom_sentiments, custom_avg_score = AnalyzeSentiment_Custom(summaries)

        daily_change = GetDailyChange(ticker)

        stock_data = {
            'Ticker': ticker,
            'Company': company_name,
            'TB_Average_Sentiment_Score': tb_avg_score,
            'Custom_Average_Sentiment_Score': custom_avg_score,
            'Daily Stock Change (%)': daily_change,
            'TB_Sentiments': tb_sentiments,
            'Custom_Sentiments': custom_sentiments
        }
        stock_data_list.append(stock_data)
    else:
        print(f"No company name found for ticker {ticker}")

# Create a DataFrame for better visualization
df_results = pd.DataFrame(stock_data_list)

try:
    # Ensure the values are numeric
    df_results['Daily Stock Change (%)'] = pd.to_numeric(df_results['Daily Stock Change (%)'], errors='coerce')
    df_results['TB_Average_Sentiment_Score'] = pd.to_numeric(df_results['TB_Average_Sentiment_Score'], errors='coerce')
    df_results['Custom_Average_Sentiment_Score'] = pd.to_numeric(df_results['Custom_Average_Sentiment_Score'], errors='coerce')
    print(df_results[['Ticker','Daily Stock Change (%)','TB_Average_Sentiment_Score','Custom_Average_Sentiment_Score']])

    # Compute correlations
    tb_correlation = df_results['Daily Stock Change (%)'].corr(df_results['TB_Average_Sentiment_Score'])
    custom_correlation = df_results['Daily Stock Change (%)'].corr(df_results['Custom_Average_Sentiment_Score'])

    print(f"Overall Correlation (Transformer-based) between Stock Change and Sentiment Score: {tb_correlation:.2f}")
    print(f"Overall Correlation (Custom Decision Tree) between Stock Change and Sentiment Score: {custom_correlation:.2f}")

except Exception as e:
    print(f"Error in calculating correlation: {e}")

# Plot the transformer-based results
bar_width = 0.3
x = np.arange(len(df_results['Ticker']))  # Position of groups

plt.figure(figsize=(12, 6))

# Create bars for daily stock change
plt.bar(x - bar_width, df_results['Daily Stock Change (%)'], bar_width, label='Daily Stock Change (%)')

# Create bars for transformer sentiment score
plt.bar(x, df_results['TB_Average_Sentiment_Score'], bar_width, label='TB Avg Sentiment Score')

# Add labels and title
plt.xticks(x, df_results['Ticker'])
plt.title('Daily Price Change vs Transformer')
plt.xlabel('Ticker')
plt.ylabel('Score/Percentage')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()
