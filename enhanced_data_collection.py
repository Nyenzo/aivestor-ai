import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from fredapi import Fred
import os
from dotenv import load_dotenv
import json
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Dict, List, Any
from transformers import pipeline
import ta

# Load environment variables
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
SENTIMENT_API_KEY = os.getenv("SENTIMENT_API_KEY")

class EnhancedDataCollector:
    def __init__(self):
        self.sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financials': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Industrials': 'XLI',
            'Utilities': 'XLU'
        }
        
        # Add sector tickers for individual stocks
        self.sector_tickers = {
            "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "ADBE"],
            "Healthcare": ["UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY"],
            "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "C"],
            "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX"],
            "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST", "MDLZ"],
            "Energy": ["XOM", "CVX", "COP", "SLB", "OXY", "MPC"],
            "Industrials": ["CAT", "BA", "HON", "UNP", "MMM", "GE"],
            "Utilities": ["NEE", "DUK", "SO", "D", "EXC", "AEP"]
        }
        
        # Enhanced FRED indicators including employment and interest rates
        self.fred_indicators = {
            'GDP': 'GDP',                           # Gross Domestic Product
            'Real_GDP': 'GDPC1',                    # Real GDP
            'Inflation': 'CPIAUCSL',                # Consumer Price Index
            'Core_Inflation': 'CPILFESL',           # Core CPI (excluding food and energy)
            'Unemployment': 'UNRATE',               # Unemployment Rate
            'Initial_Claims': 'ICSA',               # Initial Jobless Claims
            'Nonfarm_Payrolls': 'PAYEMS',          # Total Nonfarm Payrolls
            'Fed_Funds_Rate': 'FEDFUNDS',          # Federal Funds Rate
            '10Y_Treasury': 'DGS10',               # 10-Year Treasury Rate
            '2Y_Treasury': 'DGS2',                 # 2-Year Treasury Rate
            'Industrial_Production': 'INDPRO',      # Industrial Production
            'Consumer_Sentiment': 'UMCSENT',        # Consumer Sentiment
            'Retail_Sales': 'RSAFS',               # Retail Sales
            'Housing_Starts': 'HOUST',             # Housing Starts
            'PCE': 'PCE',                          # Personal Consumption Expenditures
            'Capacity_Utilization': 'TCU',         # Capacity Utilization
            'Labor_Force_Participation': 'CIVPART'  # Labor Force Participation Rate
        }
        
        if FRED_API_KEY:
            self.fred = Fred(api_key=FRED_API_KEY)
        
        self.sentiment_analyzer = pipeline('sentiment-analysis', 
                                        model='distilbert-base-uncased-finetuned-sst-2-english')
        
    def fetch_historical_market_data(self, years=5):
        """Fetch historical market data for all sector ETFs"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        market_data = {}
        for sector, ticker in self.sector_etfs.items():
            try:
                etf = yf.Ticker(ticker)
                hist = etf.history(start=start_date, end=end_date)
                
                # Calculate additional technical indicators
                hist['Returns'] = hist['Close'].pct_change()
                hist['Volatility'] = hist['Returns'].rolling(window=20).std()
                hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
                hist['RSI'] = self._calculate_rsi(hist['Close'])
                
                market_data[sector] = hist
                print(f"Successfully fetched data for {sector} ({ticker})")
                
            except Exception as e:
                print(f"Error fetching data for {sector} ({ticker}): {e}")
                
        return market_data
    
    def fetch_economic_indicators(self):
        """Fetch economic indicators from FRED"""
        if not FRED_API_KEY:
            print("FRED API key not found. Skipping economic indicators.")
            return {}
            
        economic_data = {}
        for indicator_name, series_id in self.fred_indicators.items():
            try:
                data = self.fred.get_series(series_id)
                economic_data[indicator_name] = data
                print(f"Successfully fetched {indicator_name} data")
                
            except Exception as e:
                print(f"Error fetching {indicator_name} data: {e}")
                
        return economic_data
    
    def fetch_sector_fundamentals(self):
        """Fetch fundamental data for major companies in each sector"""
        fundamentals = {}
        
        for sector, etf in self.sector_etfs.items():
            try:
                # Get ETF holdings
                etf_ticker = yf.Ticker(etf)
                holdings = etf_ticker.get_holdings()
                
                if holdings is not None and not holdings.empty:
                    top_holdings = holdings.head(5)  # Get top 5 holdings
                    sector_fundamentals = {}
                    
                    for _, holding in top_holdings.iterrows():
                        symbol = holding.name
                        stock = yf.Ticker(symbol)
                        
                        # Get key statistics
                        info = stock.info
                        fundamentals_data = {
                            'PE_Ratio': info.get('forwardPE', None),
                            'PB_Ratio': info.get('priceToBook', None),
                            'Dividend_Yield': info.get('dividendYield', None),
                            'Market_Cap': info.get('marketCap', None),
                            'Revenue_Growth': info.get('revenueGrowth', None)
                        }
                        
                        sector_fundamentals[symbol] = fundamentals_data
                    
                    fundamentals[sector] = sector_fundamentals
                    print(f"Successfully fetched fundamentals for {sector}")
                    
            except Exception as e:
                print(f"Error fetching fundamentals for {sector}: {e}")
                
        return fundamentals
    
    def _calculate_rsi(self, prices, periods=14):
        """Calculate RSI technical indicator"""
        deltas = np.diff(prices)
        seed = deltas[:periods+1]
        up = seed[seed >= 0].sum()/periods
        down = -seed[seed < 0].sum()/periods
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:periods] = 100. - 100./(1. + rs)
        
        for i in range(periods, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
                
            up = (up*(periods-1) + upval)/periods
            down = (down*(periods-1) + downval)/periods
            rs = up/down
            rsi[i] = 100. - 100./(1. + rs)
            
        return rsi
    
    def save_data(self, data: Dict[str, Any], filename: str):
        """Save collected data to file"""
        try:
            # Convert all data to JSON serializable format
            serializable_data = {}
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    serializable_data[key] = value.to_dict(orient='split')
                elif isinstance(value, pd.Series):
                    serializable_data[key] = value.to_dict()
                else:
                    serializable_data[key] = value
                    
            with open(filename, 'w') as f:
                json.dump(serializable_data, f)
            print(f"Successfully saved data to {filename}")
            
        except Exception as e:
            print(f"Error saving data to {filename}: {e}")
            
    def collect_all_data(self, tickers=None, start_date=None):
        """Collect all available data"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        if tickers is None:
            # Collect both ETFs and individual stocks
            tickers = list(self.sector_etfs.values())
            for sector_stocks in self.sector_tickers.values():
                tickers.extend(sector_stocks)

        # Collect stock data
        stock_data = {}
        for ticker in tickers:
            stock_data[ticker] = self.collect_stock_data(ticker, start_date)
            time.sleep(1)  # Avoid rate limiting
        
        # Collect economic indicators
        economic_data = self.collect_economic_data()

        # Analyze market sentiment
        market_sentiment = self.analyze_market_sentiment()

        # Save combined results
        combined_data = {
            'stock_data': stock_data,
            'economic_indicators': economic_data,
            'market_sentiment': market_sentiment,
            'collection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save to JSON
        with open('market_data.json', 'w') as f:
            json.dump({
                'collection_date': combined_data['collection_date'],
                'market_sentiment': combined_data['market_sentiment']
            }, f, indent=4)

        return combined_data

    def collect_stock_data(self, ticker, start_date):
        """Collect historical stock data with technical indicators"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date)
            
            if df.empty:
                return None
                
            # Add technical indicators
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            df['MACD'] = ta.trend.MACD(df['Close']).macd()
            df['BB_upper'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
            df['BB_lower'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
            df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
            df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
            
            # Save to CSV
            df.to_csv(f'stock_data_{ticker}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            return df
            
        except Exception as e:
            print(f"Error collecting data for {ticker}: {e}")
            return None

    def collect_economic_data(self):
        """Collect economic indicators from FRED"""
        economic_data = pd.DataFrame()
        
        for indicator_name, series_id in self.fred_indicators.items():
            try:
                series = self.fred.get_series(series_id)
                economic_data[indicator_name] = series
            except Exception as e:
                print(f"Error collecting {indicator_name}: {e}")
                
        # Save to CSV
        filename = f'economic_indicators_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        economic_data.to_csv(filename)
        return economic_data

    def analyze_market_sentiment(self, news_sources=None):
        """Analyze market sentiment from various sources"""
        if news_sources is None:
            # Use default news sources or fetch from a predefined source
            with open('sentiment_results.txt', 'r', encoding='utf-8') as file:
                news_sources = file.readlines()

        sentiments = {}
        for sector in self.sector_etfs.keys():
            sector_news = [news for news in news_sources if sector.lower() in news.lower()]
            if sector_news:
                sector_sentiments = []
                for news in sector_news:
                    try:
                        result = self.sentiment_analyzer(news[:512])[0]
                        sentiment_score = 1.0 if result['label'] == 'POSITIVE' else -1.0
                        confidence = result['score']
                        sector_sentiments.append((sentiment_score, confidence))
                    except Exception as e:
                        print(f"Error analyzing sentiment for {sector}: {e}")
                        continue
                
                if sector_sentiments:
                    # Weighted average of sentiment scores by confidence
                    weighted_sentiment = sum(score * conf for score, conf in sector_sentiments) / sum(conf for _, conf in sector_sentiments)
                    sentiments[sector] = weighted_sentiment

        return sentiments

def main():
    collector = EnhancedDataCollector()
    print("Starting enhanced data collection...")
    
    # Collect all data
    data = collector.collect_all_data()
    
    print("\nData Collection Summary:")
    print(f"Collection Date: {data['collection_date']}")
    print("\nStock Data Collected:")
    for ticker, df in data['stock_data'].items():
        if df is not None:
            print(f"- {ticker}: {len(df)} days of data")
    
    print("\nEconomic Indicators Collected:")
    if not data['economic_indicators'].empty:
        print(f"- {len(data['economic_indicators'].columns)} indicators")
        print("- Indicators:", list(data['economic_indicators'].columns))
    
    print("\nMarket Sentiment Analysis:")
    for sector, sentiment in data['market_sentiment'].items():
        print(f"- {sector}: {sentiment:.2f}")
        
if __name__ == "__main__":
    main() 