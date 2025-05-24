# Import libraries
import json
import yfinance as yf
import time  # For adding delays

# Step 1: Load Sentiment and Bill Analysis Results
def load_sentiment_results(file_path):
    sentiments = {}
    current_section = ""
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip() in ["Kenya News Sentiments:", "USA News Sentiments:", "Bill Sentiments:"]:
                current_section = line.strip().replace(" Sentiments:", "")
            elif ":" in line and current_section:
                text, sentiment_conf = line.split(" | Sentiment: ")
                sentiment, confidence = sentiment_conf.split(" | Confidence: ")
                confidence = float(confidence.strip())
                sentiments[f"{current_section} | {text.replace('Text: ', '').strip()}"] = {
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "section": current_section
                }
    return sentiments

def load_bill_analysis_results(file_path):
    bills = []
    current_bill = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith("Bill"):
                if current_bill:
                    bills.append(current_bill)
                current_bill = {"text": line.split(": ")[1].split("...")[0] + "...", "key_terms": [], "sector_impacts": {}}
            elif line.startswith("Key Terms:"):
                current_bill["key_terms"] = line.replace("Key Terms: ", "").split(", ")
            elif line.startswith("  "):
                sector = line.strip()
                impacts = {}
                for next_line in file:
                    if next_line.strip().startswith("  "):
                        break
                    if "    - " in next_line:
                        term, impact = next_line.replace("    - ", "").split(": ")
                        impacts[term] = impact
                current_bill["sector_impacts"][sector] = impacts
        if current_bill:
            bills.append(current_bill)
    return bills

# Step 2: Combine Analysis
def combine_analysis(sentiments, bills):
    combined_results = []

    # Process bills (Kenya-specific, but will skip stock data for now)
    for bill in bills:
        bill_text = bill["text"]
        key = f"Bill | {bill_text}"
        if key in sentiments:
            sentiment_data = sentiments[key]
            combined_results.append({
                "text": bill_text,
                "sentiment": sentiment_data["sentiment"],
                "confidence": sentiment_data["confidence"],
                "key_terms": bill["key_terms"],
                "sector_impacts": bill["sector_impacts"],
                "market": "Kenya",
                "type": "bill"
            })

    # Process Kenya news (will skip stock data for now)
    kenya_news = [item for key, item in sentiments.items() if key.startswith("Kenya News |")]
    if kenya_news:
        avg_confidence = sum(item["confidence"] for item in kenya_news) / len(kenya_news)
        positive_count = sum(1 for item in kenya_news if item["sentiment"] == "POSITIVE")
        negative_count = sum(1 for item in kenya_news if item["sentiment"] == "NEGATIVE")
        overall_sentiment = "POSITIVE" if positive_count > negative_count else "NEGATIVE" if negative_count > positive_count else "NEUTRAL"
        combined_results.append({
            "text": "Kenya News Aggregate",
            "sentiment": overall_sentiment,
            "confidence": avg_confidence,
            "key_terms": [],
            "sector_impacts": {"General Business": {"news": "General Market Sentiment"}},
            "market": "Kenya",
            "type": "news"
        })

    # Process USA news (apply sentiment to all sectors)
    usa_news = [item for key, item in sentiments.items() if key.startswith("USA News |")]
    if usa_news:
        avg_confidence = sum(item["confidence"] for item in usa_news) / len(usa_news)
        positive_count = sum(1 for item in usa_news if item["sentiment"] == "POSITIVE")
        negative_count = sum(1 for item in usa_news if item["sentiment"] == "NEGATIVE")
        overall_sentiment = "POSITIVE" if positive_count > negative_count else "NEGATIVE" if negative_count > positive_count else "NEUTRAL"
        # Define all sectors for USA news
        sector_impacts = {
            "Technology": {"news": "General Market Sentiment"},
            "Healthcare": {"news": "General Market Sentiment"},
            "Financials": {"news": "General Market Sentiment"},
            "Consumer Discretionary": {"news": "General Market Sentiment"},
            "Consumer Staples": {"news": "General Market Sentiment"},
            "Energy": {"news": "General Market Sentiment"},
            "Industrials": {"news": "General Market Sentiment"},
            "Utilities": {"news": "General Market Sentiment"}
        }
        combined_results.append({
            "text": "USA News Aggregate",
            "sentiment": overall_sentiment,
            "confidence": avg_confidence,
            "key_terms": [],
            "sector_impacts": sector_impacts,
            "market": "USA",
            "type": "news"
        })

    return combined_results

# Step 3: Fetch Stock Data with yfinance
def get_stock_data(ticker, short_term_period="3mo", long_term_period="1y"):
    try:
        stock = yf.Ticker(ticker)
        short_term_hist = stock.history(period=short_term_period)
        long_term_hist = stock.history(period=long_term_period)
        
        short_term_data = {"price": None, "change": None}
        long_term_data = {"price": None, "change": None}
        
        if not short_term_hist.empty:
            short_term_data["price"] = round(float(short_term_hist["Close"].iloc[-1]), 2)
            short_term_data["change"] = round(((short_term_hist["Close"].iloc[-1] - short_term_hist["Close"].iloc[0]) / short_term_hist["Close"].iloc[0]) * 100, 2)
        
        if not long_term_hist.empty:
            long_term_data["price"] = round(float(long_term_hist["Close"].iloc[-1]), 2)
            long_term_data["change"] = round(((long_term_hist["Close"].iloc[-1] - long_term_hist["Close"].iloc[0]) / long_term_hist["Close"].iloc[0]) * 100, 2)
        
        # Add a delay to avoid rate limiting
        time.sleep(1)  # 1-second delay between requests
        
        return {"short_term": short_term_data, "long_term": long_term_data}
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return {"short_term": {"price": "N/A", "change": "N/A"}, "long_term": {"price": "N/A", "change": "N/A"}}

# Step 4: Predict Stock Impact
def predict_stock_impact(combined_result):
    sentiment = combined_result["sentiment"]
    confidence = combined_result["confidence"]
    sector_impacts = combined_result["sector_impacts"]
    market = combined_result["market"]

    # Define sector-to-ticker mappings (USA only for now, Kenya skipped)
    sector_tickers = {
        "USA": {
            "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "ADBE"],
            "Healthcare": ["UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY"],
            "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "C"],
            "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX"],
            "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST", "MDLZ"],
            "Energy": ["XOM", "CVX", "COP", "SLB", "OXY", "MPC"],
            "Industrials": ["CAT", "BA", "HON", "UNP", "MMM", "GE"],
            "Utilities": ["NEE", "DUK", "SO", "D", "EXC", "AEP"]
        }
    }

    # Initialize predictions for the relevant market only
    predictions = {
        market: {
            "short_term_impact": {"prediction": "Neutral", "reason": "No strong indicators"},
            "long_term_impact": {"prediction": "Neutral", "reason": "No strong indicators"},
            "stock_data": {}
        }
    }

    # Short-term prediction (1-3 months)
    if confidence > 0.9:
        if sentiment == "POSITIVE":
            predictions[market]["short_term_impact"]["prediction"] = "Bullish"
            predictions[market]["short_term_impact"]["reason"] = "High positive sentiment likely to boost investor confidence quickly"
        elif sentiment == "NEGATIVE":
            predictions[market]["short_term_impact"]["prediction"] = "Bearish"
            predictions[market]["short_term_impact"]["reason"] = "High negative sentiment may lead to immediate market reaction"

    # Long-term prediction (4-12 months)
    for sector, impacts in sector_impacts.items():
        if any("Potential Regulatory Change" in i for i in impacts.values()):
            predictions[market]["long_term_impact"]["reason"] = "Regulatory changes may have lasting effects"
            predictions[market]["long_term_impact"]["prediction"] = "Uncertain" if sentiment == "NEUTRAL" else "Bearish" if sentiment == "NEGATIVE" else "Bullish"
        if any("Potential Cost Increase" in i for i in impacts.values()):
            predictions[market]["long_term_impact"]["reason"] = "Cost increases may impact profitability over time"
            predictions[market]["long_term_impact"]["prediction"] = "Bearish"

    # Fetch stock data for relevant sectors
    stock_data = {}
    for sector in sector_impacts.keys():
        # Get list of tickers for the sector, default to empty list if sector not found
        tickers = sector_tickers.get(market, {}).get(sector, [])
        stock_data[sector] = {}
        for ticker in tickers:
            stock_data[sector][ticker] = get_stock_data(ticker)
    predictions[market]["stock_data"] = stock_data

    return predictions

# Step 5: Main Function
def main():
    # Load results
    sentiment_results = load_sentiment_results("sentiment_results.txt")
    bill_analysis_results = load_bill_analysis_results("bill_analysis_results.txt")
    
    # Combine results
    combined_results = combine_analysis(sentiment_results, bill_analysis_results)
    
    # Print and save
    print("Combined Analysis Results:")
    for result in combined_results:
        print(f"\nText: {result['text']}")
        print(f"Type: {result['type'].capitalize()}")
        print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f})")
        print("Key Terms:", result['key_terms'] if result['key_terms'] else "N/A")
        print("Sector Impacts:")
        for sector, impacts in result["sector_impacts"].items():
            print(f"  {sector}:")
            for term, impact in impacts.items():
                print(f"    - {term}: {impact}")
        # Predict stock impact for the relevant market
        impact = predict_stock_impact(result)
        market = result["market"]
        print(f"\n{market} Stock Impact Prediction:")
        print("  Short-Term (1-3 Months):")
        print(f"    Prediction: {impact[market]['short_term_impact']['prediction']}")
        print(f"    Reason: {impact[market]['short_term_impact']['reason']}")
        print("  Long-Term (4-12 Months):")
        print(f"    Prediction: {impact[market]['long_term_impact']['prediction']}")
        print(f"    Reason: {impact[market]['long_term_impact']['reason']}")
        for sector, ticker_data in impact[market]["stock_data"].items():
            print(f"  {sector} Stock Data:")
            for ticker, data in ticker_data.items():
                print(f"    {ticker}:")
                short_price = data['short_term']['price'] if data['short_term']['price'] is not None else "N/A"
                short_change = data['short_term']['change'] if data['short_term']['change'] is not None else "N/A"
                long_price = data['long_term']['price'] if data['long_term']['price'] is not None else "N/A"
                long_change = data['long_term']['change'] if data['long_term']['change'] is not None else "N/A"
                print(f"      Short-Term: Price={short_price}, Change={short_change}%")
                print(f"      Long-Term: Price={long_price}, Change={long_change}%")

    # Save to file
    with open("combined_analysis_results.txt", "w", encoding="utf-8") as file:
        for result in combined_results:
            file.write(f"\nText: {result['text']}\n")
            file.write(f"Type: {result['type'].capitalize()}\n")
            file.write(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2f})\n")
            file.write("Key Terms: " + (", ".join(result['key_terms']) if result['key_terms'] else "N/A") + "\n")
            file.write("Sector Impacts:\n")
            for sector, impacts in result["sector_impacts"].items():
                file.write(f"  {sector}:\n")
                for term, impact in impacts.items():
                    file.write(f"    - {term}: {impact}\n")
            impact = predict_stock_impact(result)
            market = result["market"]
            file.write(f"\n{market} Stock Impact Prediction:\n")
            file.write("  Short-Term (1-3 Months):\n")
            file.write(f"    Prediction: {impact[market]['short_term_impact']['prediction']}\n")
            file.write(f"    Reason: {impact[market]['short_term_impact']['reason']}\n")
            file.write("  Long-Term (4-12 Months):\n")
            file.write(f"    Prediction: {impact[market]['long_term_impact']['prediction']}\n")
            file.write(f"    Reason: {impact[market]['long_term_impact']['reason']}\n")
            for sector, ticker_data in impact[market]["stock_data"].items():
                file.write(f"  {sector} Stock Data:\n")
                for ticker, data in ticker_data.items():
                    file.write(f"    {ticker}:\n")
                    short_price = data['short_term']['price'] if data['short_term']['price'] is not None else "N/A"
                    short_change = data['short_term']['change'] if data['short_term']['change'] is not None else "N/A"
                    long_price = data['long_term']['price'] if data['long_term']['price'] is not None else "N/A"
                    long_change = data['long_term']['change'] if data['long_term']['change'] is not None else "N/A"
                    file.write(f"      Short-Term: Price={short_price}, Change={short_change}%\n")
                    file.write(f"      Long-Term: Price={long_price}, Change={long_change}%\n")
    print("\nCombined analysis results saved to combined_analysis_results.txt")

if __name__ == "__main__":
    main()