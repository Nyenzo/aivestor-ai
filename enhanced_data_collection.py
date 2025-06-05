# Collect comprehensive market data for all defined sectors and tickers
import yfinance as yf
import pandas as pd
import os
import json
from datetime import datetime, timedelta
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange
from newsapi import NewsApiClient
from transformers import pipeline
import requests

class EnhancedDataCollector:
    def __init__(self):
        # Initialize API clients with environment variables
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.newsapi_key = os.getenv('NEWSAPI_KEY')
        self.ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas') if self.alpha_vantage_key else None
        self.fx = ForeignExchange(key=self.alpha_vantage_key) if self.alpha_vantage_key else None
        self.newsapi = NewsApiClient(api_key=self.newsapi_key) if self.newsapi_key else None
        self.sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

    def collect_stock_data(self, ticker: str, start_date: str) -> pd.DataFrame:
        # Fetch stock price data with caching to respect API limits
        cache_file = f'stock_data_{ticker}.json'
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            cache_date = datetime.strptime(cached_data['timestamp'], '%Y-%m-%d %H:%M:%S')
            if cache_date > datetime.now() - timedelta(days=1):
                print(f"Using cached data for {ticker}")
                return pd.DataFrame(cached_data['data'])
        
        df = yf.Ticker(ticker).history(start=start_date)
        if not df.empty:
            df['RSI'] = self.calculate_rsi(df['Close'])
            df['MACD'] = self.calculate_macd(df['Close'])
            df['BB_upper'], df['BB_lower'] = self.calculate_bollinger_bands(df['Close'])
            df['ATR'] = self.calculate_atr(df)
            df = df.reset_index()
            cache_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data': df.to_dict(orient='records')
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        return df

    def collect_xauusd_data(self, start_date: str) -> pd.DataFrame:
        # Fetch XAUUSD spot price using Alpha Vantage with caching
        cache_file = 'xauusd_data.json'
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            cache_date = datetime.strptime(cached_data['timestamp'], '%Y-%m-%d %H:%M:%S')
            if cache_date > datetime.now() - timedelta(days=1):
                print("Using cached XAUUSD data")
                return pd.DataFrame(cached_data['data'])
        
        if not self.fx:
            print("Alpha Vantage API key missing, using default data")
            return pd.DataFrame({'Date': pd.date_range(start=start_date, periods=365, freq='D'), 'Close': [1800]*365})
        
        try:
            data, _ = self.fx.get_currency_exchange_daily(from_symbol='XAU', to_symbol='USD', outputsize='full')
            df = data.rename(columns={'4. close': 'Close'})
            df.index = pd.to_datetime(df.index)
            df = df[df.index >= start_date][['Close']].astype(float)
            df['RSI'] = self.calculate_rsi(df['Close'])
            df['MACD'] = self.calculate_macd(df['Close'])
            df['BB_upper'], df['BB_lower'] = self.calculate_bollinger_bands(df['Close'])
            df['ATR'] = self.calculate_atr(df)
            df = df.reset_index()
            cache_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data': df.to_dict(orient='records')
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            return df
        except Exception as e:
            print(f"Error fetching XAUUSD: {e}")
            return pd.DataFrame({'Date': pd.date_range(start=start_date, periods=365, freq='D'), 'Close': [1800]*365})

    def collect_economic_indicators(self) -> pd.DataFrame:
        # Fetch economic indicators (simplified for demo)
        cache_file = 'economic_indicators.json'
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            cache_date = datetime.strptime(cached_data['timestamp'], '%Y-%m-%d %H:%M:%S')
            if cache_date > datetime.now() - timedelta(days=1):
                print("Using cached economic indicators")
                return pd.DataFrame(cached_data['data'])
        
        df = pd.DataFrame({
            'Date': pd.date_range(start='2018-01-01', periods=365, freq='D'),
            'GDP': [2.5]*365,
            'Inflation': [2.0]*365,
            'Unemployment': [4.0]*365,
            'FedFundsRate': [1.5]*365
        })
        cache_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data': df.to_dict(orient='records')
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        return df

    def collect_sector_news(self, sector: str) -> List[Dict]:
        # Fetch and analyze news for sentiment
        cache_file = f'sector_news_{sector}.json'
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            cache_date = datetime.strptime(cached_data['timestamp'], '%Y-%m-%d %H:%M:%S')
            if cache_date > datetime.now() - timedelta(days=1):
                print(f"Using cached news for {sector}")
                return cached_data['data']
        
        if not self.newsapi:
            return []
        
        articles = self.newsapi.get_everything(q=sector, language='en', page_size=20)
        sentiments = []
        for article in articles.get('articles', []):
            sentiment = self.sentiment_analyzer(article['description'] or article['title'])[0]
            sentiments.append({
                'title': article['title'],
                'sentiment': sentiment['label'],
                'score': sentiment['score']
            })
        cache_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data': sentiments
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        return sentiments

    def collect_all_data(self, start_date: str = '2018-01-01', tickers: List[str] = None) -> Dict:
        # Collect all data types for specified tickers or all sectors
        stock_data = {}
        from advanced_stock_predictor import AdvancedStockPredictor
        predictor = AdvancedStockPredictor()
        
        target_tickers = tickers if tickers else predictor.get_all_tickers()
        for ticker in target_tickers:
            if ticker == 'XAUUSD':
                stock_data[ticker] = self.collect_xauusd_data(start_date)
            else:
                stock_data[ticker] = self.collect_stock_data(ticker, start_date)
        
        economic_data = self.collect_economic_indicators()
        sector_sentiments = {sector: self.collect_sector_news(sector) 
                           for sector in predictor.sector_etfs.keys()}
        
        return {
            'stock_data': stock_data,
            'economic_data': economic_data,
            'market_sentiment': sector_sentiments
        }

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        # Calculate Relative Strength Index
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        # Calculate MACD
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        return macd

    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: int = 2) -> tuple:
        # Calculate Bollinger Bands
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, lower

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        # Calculate Average True Range
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()