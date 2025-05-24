# Import libraries for sentiment analysis and data handling
import pandas as pd
from transformers import pipeline
import torch

# Step 1: Load the processed data from processed_data.txt
def load_processed_data(file_path):
    """
    Load news, USA news, and bills from the processed data file.
    Args:
        file_path (str): Path to the processed_data.txt file
    Returns:
        tuple: Three lists (news, usa_news, bills)
    """
    news = []
    usa_news = []
    bills = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith("NEWS:"):
                news.append(line.replace("NEWS:", "").strip())
            elif line.startswith("USA_NEWS:"):
                usa_news.append(line.replace("USA_NEWS:", "").strip())
            elif line.startswith("BILL:"):
                bills.append(line.replace("BILL:", "").strip())
    return news, usa_news, bills

# Step 2: Perform Sentiment Analysis using BERT
def analyze_sentiment(texts, classifier):
    """
    Analyze sentiment of a list of texts using BERT.
    Args:
        texts (list): List of text strings to analyze
        classifier: Pre-trained BERT sentiment analysis pipeline
    Returns:
        list: List of (text, sentiment, confidence) tuples
    """
    results = []
    for text in texts:
        text = text[:512] if len(text) > 512 else text
        try:
            result = classifier(text)[0]
            sentiment = result['label']
            confidence = result['score']
            results.append((text, sentiment, confidence))
        except Exception as e:
            print(f"Error analyzing text: {text[:50]}... Error: {e}")
            results.append((text, "ERROR", 0.0))
    return results

# Step 3: Main Function to Run Sentiment Analysis
def main():
    # Load pre-trained BERT sentiment analysis model
    print("Loading BERT sentiment analysis model...")
    classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    print("Model loaded successfully.")

    # Load the processed data
    news, usa_news, bills = load_processed_data("processed_data.txt")
    print(f"Loaded {len(news)} Kenya news articles, {len(usa_news)} USA news articles, and {len(bills)} bills.")

    # Analyze sentiment for Kenya news
    print("\nAnalyzing sentiment for Kenya news articles...")
    news_sentiments = analyze_sentiment(news, classifier)
    for i, (text, sentiment, confidence) in enumerate(news_sentiments):
        print(f"News {i + 1}: {text[:50]}... | Sentiment: {sentiment} | Confidence: {confidence:.2f}")

    # Analyze sentiment for USA news
    print("\nAnalyzing sentiment for USA news articles...")
    usa_news_sentiments = analyze_sentiment(usa_news, classifier)
    for i, (text, sentiment, confidence) in enumerate(usa_news_sentiments):
        print(f"USA News {i + 1}: {text[:50]}... | Sentiment: {sentiment} | Confidence: {confidence:.2f}")

    # Analyze sentiment for bills
    print("\nAnalyzing sentiment for bills...")
    bill_sentiments = analyze_sentiment(bills, classifier)
    for i, (text, sentiment, confidence) in enumerate(bill_sentiments):
        print(f"Bill {i + 1}: {text[:50]}... | Sentiment: {sentiment} | Confidence: {confidence:.2f}")

    # Save results to a file
    with open("sentiment_results.txt", "w", encoding="utf-8") as file:
        file.write("Kenya News Sentiments:\n")
        for text, sentiment, confidence in news_sentiments:
            file.write(f"Text: {text[:50]}... | Sentiment: {sentiment} | Confidence: {confidence:.2f}\n")
        file.write("\nUSA News Sentiments:\n")
        for text, sentiment, confidence in usa_news_sentiments:
            file.write(f"Text: {text[:50]}... | Sentiment: {sentiment} | Confidence: {confidence:.2f}\n")
        file.write("\nBill Sentiments:\n")
        for text, sentiment, confidence in bill_sentiments:
            file.write(f"Text: {text[:50]}... | Sentiment: {sentiment} | Confidence: {confidence:.2f}\n")
    print("Sentiment analysis results saved to sentiment_results.txt")

if __name__ == "__main__":
    main()