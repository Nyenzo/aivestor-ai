# Import libraries for data collection, preprocessing, and PDF handling
import requests
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import datetime
import PyPDF2
import os
from pdf2image import convert_from_path
import pytesseract

# Step 1: Preprocess Text Data (for news and bills)
def preprocess_text(text):
    """
    Preprocess text by tokenizing, removing stopwords, and cleaning.
    Args:
        text (str): Raw text to preprocess
    Returns:
        str: Processed text
    """
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(tokens)

# Step 2: Fetch Real-Time News Articles using NewsAPI (Kenya)
def fetch_news(api_key, query="Kenya stock"):
    """
    Fetch real-time news articles using NewsAPI for Kenya.
    Args:
        api_key (str): Your NewsAPI key
        query (str): Search query for news
    Returns:
        list: List of news articles with titles and descriptions
    """
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        data = response.json()
        articles = data['articles']
        return [(article['title'], article['description']) for article in articles if article['description']]
    else:
        print(f"Error fetching Kenya news: {response.status_code}")
        return []

# Step 2.5: Fetch Real-Time News Articles using NewsAPI (USA)
def fetch_usa_news(api_key):
    """
    Fetch real-time USA news articles using NewsAPI.
    Args:
        api_key (str): Your NewsAPI key
    Returns:
        list: List of news articles with titles and descriptions
    """
    url = f"https://newsapi.org/v2/everything?q=USA+stock+market&apiKey={api_key}"
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        data = response.json()
        articles = data['articles']
        return [(article['title'], article['description']) for article in articles if article['description']]
    else:
        print(f"Error fetching USA news: {response.status_code}")
        return []

# Step 3: Extract Text from PDF Files (with OCR fallback, no page limit)
def extract_text_from_pdf(pdf_path):
    """
    Extract text from a single PDF file, using OCR if needed.
    Args:
        pdf_path (str): Path to the PDF file
    Returns:
        str: Extracted text from the entire PDF
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):  # Process all pages
                page = reader.pages[page_num]
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
            if text and not text.isspace():
                return text
            else:
                print(f"PyPDF2 failed for {pdf_path}, falling back to OCR...")
    except Exception as e:
        print(f"PyPDF2 error for {pdf_path}: {e}, falling back to OCR...")

    try:
        images = convert_from_path(pdf_path, dpi=200)  # Process all pages
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image) + " "
        return text.strip()
    except Exception as e:
        print(f"OCR error for {pdf_path}: {e}")
        return ""

# Step 4: Load Government Bills from PDF Files
def load_bills_from_pdfs(directory):
    """
    Load and extract text from all PDF files in the specified directory.
    Args:
        directory (str): Directory containing PDF files
    Returns:
        list: List of extracted texts from PDFs
    """
    bills = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            print(f"Extracting text from {filename}...")
            text = extract_text_from_pdf(pdf_path)
            processed_text = preprocess_text(text)
            if processed_text:
                bills.append(processed_text)
    return bills

# Step 5: Main Function to Collect Data
def main():
    # Fetch news articles for Kenya and USA
    api_key = "9e6d891c6a8745d4b78e964a43921748"
    news_articles = fetch_news(api_key)
    usa_news_articles = fetch_usa_news(api_key)
    print(f"Fetched {len(news_articles)} Kenya news articles and {len(usa_news_articles)} USA news articles.")

    # Preprocess news
    processed_news = [preprocess_text(title + ' ' + desc) for title, desc in news_articles]
    processed_usa_news = [preprocess_text(title + ' ' + desc) for title, desc in usa_news_articles]

    # Load and preprocess bills
    bills = load_bills_from_pdfs("bills")
    print(f"Loaded {len(bills)} government bills.")

    # Save to processed_data.txt
    with open("processed_data.txt", "w", encoding="utf-8") as file:
        for news_text in processed_news:
            file.write(f"NEWS: {news_text}\n")
        for usa_news_text in processed_usa_news:
            file.write(f"USA_NEWS: {usa_news_text}\n")
        for bill_text in bills:
            file.write(f"BILL: {bill_text}\n")
    print("Processed data saved to processed_data.txt")

if __name__ == "__main__":
    print("Starting data collection...")
    main()