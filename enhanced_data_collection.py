# Collect comprehensive market data including VIX, fundamentals, FRED indicators, and XAUUSD
import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime, timedelta
from typing import List, Dict
from transformers import pipeline
import time
from dotenv import load_dotenv
from fredapi import Fred

class EnhancedDataCollector:
    DATA_DIR = 'datacollection'

    def __init__(self):
        # Load .env file
        load_dotenv()
        
        # Create data directory
        os.makedirs(self.DATA_DIR, exist_ok=True)

        # Debug environment variables
        print("ALPHA_VANTAGE_API_KEY:", os.getenv('ALPHA_VANTAGE_API_KEY'))
        print("FRED_API_KEY:", os.getenv('FRED_API_KEY'))
        print("NEWSAPI_KEY:", os.getenv('NEWSAPI_KEY'))
        
        # Initialize FRED client with environment variable
        self.fred_api_key = os.getenv('FRED_API_KEY')
        self.sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
        
        # FRED economic indicators
        self.fred_indicators = {
            'GDP': 'GDP',
            'Real_GDP': 'GDPC1',
            'Inflation': 'CPIAUCSL',
            'Core_Inflation': 'CPILFESL',
            'Unemployment': 'UNRATE',
            'Initial_Claims': 'ICSA',
            'Nonfarm_Payrolls': 'PAYEMS',
            'Fed_Funds_Rate': 'FEDFUNDS',
            '10Y_Treasury': 'DGS10',
            '2Y_Treasury': 'DGS2',
            'Industrial_Production': 'INDPRO',
            'Consumer_Sentiment': 'UMCSENT',
            'Retail_Sales': 'RSAFS',
            'Housing_Starts': 'HOUST',
            'PCE': 'PCE',
            'Capacity_Utilization': 'TCU',
            'Labor_Force_Participation': 'CIVPART'
        }

    def collect_stock_data(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        # Fetch stock price data with caching
        cache_file = os.path.join(self.DATA_DIR, f'stock_data_{ticker}.json')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                cache_date = datetime.strptime(cached_data['timestamp'], '%Y-%m-%d %H:%M:%S')
                if cache_date > datetime.now() - timedelta(days=1):
                    print(f"Using cached data for {ticker}")
                    df = pd.DataFrame(cached_data['data'])
                    df['Date'] = pd.to_datetime(df['Date'])  # Convert back to datetime
                    return df
            except Exception as e:
                print(f"Error loading cache for {ticker}: {e}")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df = yf.Ticker(ticker).history(start=start_date, end=end_date or datetime.now().strftime('%Y-%m-%d'))
                if df.empty or 'Date' not in df.columns:
                    print(f"No data for {ticker}, returning empty DataFrame")
                    return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'ATR'])
                df['RSI'] = self.calculate_rsi(df['Close'])
                df['MACD'] = self.calculate_macd(df['Close'])
                df['BB_upper'], df['BB_lower'] = self.calculate_bollinger_bands(df['Close'])
                df['ATR'] = self.calculate_atr(df)
                df = df.reset_index()
                df['Date'] = df['Date'].astype(str)  # Convert Timestamp to string for JSON
                cache_data = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%M:%S'),
                    'data': df.to_dict(orient='records')
                }
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f)
                df['Date'] = pd.to_datetime(df['Date'])  # Convert back to datetime for return
                return df
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)  # Wait 5 seconds before retry
                else:
                    print(f"Max retries reached for {ticker}, returning empty DataFrame")
                    return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'ATR'])

    def collect_xauusd_data(self, start_date: str, end_date: str = None) -> pd.DataFrame:
        # Use GLD (Gold ETF) as a proxy for XAUUSD since yfinance doesn't provide forex directly
        ticker = 'GLD'
        cache_file = os.path.join(self.DATA_DIR, 'xauusd_data.json')
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            cache_date = datetime.strptime(cached_data['timestamp'], '%Y-%m-%d %H:%M:%S')
            if cache_date > datetime.now() - timedelta(days=1):
                print("Using cached XAUUSD data")
                df = pd.DataFrame(cached_data['data'])
                df['Date'] = pd.to_datetime(df['Date'])
                return df
        
        try:
            df = yf.Ticker(ticker).history(start=start_date, end=end_date or datetime.now().strftime('%Y-%m-%d'))
            if df.empty or 'Date' not in df.columns:
                print(f"No data for {ticker}, returning default XAUUSD data")
                dates = pd.date_range(start=start_date, end=end_date or datetime.now(), freq='D')
                return pd.DataFrame({
                    'Date': dates,
                    'Close': [1800.0]*len(dates),
                    'Open': [1800.0]*len(dates),
                    'High': [1800.0]*len(dates),
                    'Low': [1800.0]*len(dates)
                })
            df['RSI'] = self.calculate_rsi(df['Close'])
            df['MACD'] = self.calculate_macd(df['Close'])
            df['BB_upper'], df['BB_lower'] = self.calculate_bollinger_bands(df['Close'])
            df['ATR'] = self.calculate_atr(df)
            df = df.reset_index()
            df['Date'] = df['Date'].astype(str)
            cache_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data': df.to_dict(orient='records')
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            print(f"Error fetching XAUUSD data via {ticker}: {e}")
            dates = pd.date_range(start=start_date, end=end_date or datetime.now(), freq='D')
            return pd.DataFrame({
                'Date': dates,
                'Close': [1800.0]*len(dates),
                'Open': [1800.0]*len(dates),
                'High': [1800.0]*len(dates),
                'Low': [1800.0]*len(dates)
            })

    def collect_vix_data(self, start_date: str, end_date: str = None) -> pd.DataFrame:
        cache_file = os.path.join(self.DATA_DIR, 'vix_data.csv')
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file)
                cache_date = pd.to_datetime(df['timestamp'].iloc[0])
                if cache_date > datetime.now() - timedelta(days=1):
                    print("Using cached VIX data")
                    df['Date'] = pd.to_datetime(df['Date'])
                    return df.drop(columns=['timestamp'])
            except Exception:
                print("Invalid VIX cache file, fetching new data")
        
        try:
            vix = yf.Ticker('^VIX').history(start=start_date, end=end_date or datetime.now().strftime('%Y-%m-%d'))
            if not vix.empty:
                vix_data = vix[['Close']].rename(columns={'Close': 'VIX'})
                vix_data = vix_data.reset_index()
                vix_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                vix_data.to_csv(cache_file, index=False)
                return vix_data.drop(columns=['timestamp'])
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching VIX data: {e}")
            return pd.DataFrame()

    def collect_fundamental_data(self, ticker: str) -> Dict:
        cache_file = os.path.join(self.DATA_DIR, f'fundamental_data_{ticker}.json')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                cache_date = datetime.strptime(cached_data['timestamp'], '%Y-%m-%d %H:%M:%S')
                if cache_date > datetime.now() - timedelta(days=7):
                    print(f"Using cached fundamental data for {ticker}")
                    return cached_data['data']
            except Exception:
                print(f"Invalid cache file for {ticker}, fetching new data")
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            fundamentals = {
                'PE_Ratio': float(info.get('trailingPE', 15.0)) if info.get('trailingPE') else 15.0,
                'EPS': float(info.get('trailingEps', 5.0)) if info.get('trailingEps') else 5.0,
                'Revenue_TTM': float(info.get('totalRevenue', 1000000000.0)) if info.get('totalRevenue') else 1000000000.0,
                'Debt_to_Equity': float(info.get('debtToEquity', 1.0)) if info.get('debtToEquity') else 1.0
            }
            cache_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data': fundamentals
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            return fundamentals
        except Exception as e:
            print(f"Error fetching fundamentals for {ticker} via yfinance: {e}")
            return {
                'PE_Ratio': 15.0,
                'EPS': 5.0,
                'Revenue_TTM': 1000000000.0,
                'Debt_to_Equity': 1.0
            }

    def collect_fred_data(self, start_date: str, end_date: str = None) -> pd.DataFrame:
        cache_file = os.path.join(self.DATA_DIR, 'fred_data.csv')
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file)
                cache_date = pd.to_datetime(df['timestamp'].iloc[0])
                if cache_date > datetime.now() - timedelta(days=1):
                    print("Using cached FRED data")
                    df['Date'] = pd.to_datetime(df['Date'])
                    return df.drop(columns=['timestamp'])
            except Exception:
                print("Invalid FRED cache file, fetching new data")
        
        fred = Fred(api_key=os.getenv('FRED_API_KEY')) if os.getenv('FRED_API_KEY') else None
        if not fred:
            print("FRED API key missing, using default data")
            default_data = {'Date': pd.date_range(start=start_date, end=end_date or datetime.now(), freq='D')}
            default_data.update({name: [0.0]*len(default_data['Date']) for name in self.fred_indicators})
            return pd.DataFrame(default_data)
        
        economic_data = {}
        for name, series_id in self.fred_indicators.items():
            try:
                print(f"Collecting {name} from FRED...")
                series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
                economic_data[name] = series
                time.sleep(0.5)
            except Exception as e:
                print(f"Error collecting {name}: {e}")
                economic_data[name] = pd.Series([0.0]*len(pd.date_range(start=start_date, end=end_date or datetime.now(), freq='D')), 
                                               index=pd.date_range(start=start_date, end=end_date or datetime.now(), freq='D'))
        
        if 'Fed_Funds_Rate' in economic_data and '10Y_Treasury' in economic_data:
            economic_data['Yield_Curve_Spread'] = economic_data['10Y_Treasury'] - economic_data['Fed_Funds_Rate']
        if 'GDP' in economic_data:
            economic_data['GDP_Growth'] = economic_data['GDP'].pct_change()
        if 'Nonfarm_Payrolls' in economic_data:
            economic_data['Employment_Change'] = economic_data['Nonfarm_Payrolls'].pct_change()
        
        df = pd.DataFrame(economic_data)
        df.index.name = 'Date'
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        
        df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df.to_csv(cache_file, index=False)
        return df.drop(columns=['timestamp'])

    def collect_sector_news(self, sector: str) -> List[Dict]:
        cache_file = os.path.join(self.DATA_DIR, f'sector_news_{sector}.json')
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            cache_date = datetime.strptime(cached_data['timestamp'], '%Y-%m-%d %H:%M:%S')
            if cache_date > datetime.now() - timedelta(days=1):
                print(f"Using cached news for {sector}")
                return cached_data['data']
        
        from newsapi import NewsApiClient
        newsapi = NewsApiClient(api_key=os.getenv('NEWSAPI_KEY')) if os.getenv('NEWSAPI_KEY') else None
        if not newsapi:
            print(f"NewsAPI key missing for {sector}, returning mock data")
            return [{'title': f'Mock news for {sector}', 'sentiment': 'NEUTRAL', 'score': 0.5}]
        
        try:
            articles = newsapi.get_everything(q=sector, language='en', page_size=20)
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
        except Exception as e:
            print(f"Error fetching news for {sector}: {e}")
            return [{'title': f'Mock news for {sector}', 'sentiment': 'NEUTRAL', 'score': 0.5}]

    def collect_all_data(self, start_date: str = '2018-01-01', end_date: str = None, tickers: List[str] = None) -> Dict:
        print(f"Starting data collection for {tickers if tickers else 'all tickers'} from {start_date} to {end_date or 'present'}")
        stock_data = {}
        fundamental_data = {}
        from advanced_stock_predictor import AdvancedStockPredictor
        predictor = AdvancedStockPredictor()
        
        target_tickers = tickers if tickers else predictor.get_all_tickers()
        print(f"Collecting data for {len(target_tickers)} tickers")
        for ticker in target_tickers:
            try:
                if ticker in ['TOT', 'PXD', 'RDS.A', 'PSA']:
                    print(f"Skipping delisted ticker: {ticker}")
                    continue
                if ticker == 'XAUUSD':
                    stock_data[ticker] = self.collect_xauusd_data(start_date, end_date)
                else:
                    stock_data[ticker] = self.collect_stock_data(ticker, start_date, end_date)
                    if ticker not in predictor.sector_etfs.values():
                        fundamental_data[ticker] = self.collect_fundamental_data(ticker)
            except Exception as e:
                print(f"Error collecting data for {ticker}: {e}")
                stock_data[ticker] = pd.DataFrame()
        
        economic_data = self.collect_fred_data(start_date, end_date)
        vix_data = self.collect_vix_data(start_date, end_date)
        sector_sentiments = {sector: self.collect_sector_news(sector) 
                           for sector in predictor.sector_etfs.keys()}
        
        print("Data collection complete")
        return {
            'stock_data': stock_data,
            'fundamental_data': fundamental_data,
            'economic_data': economic_data,
            'vix_data': vix_data,
            'market_sentiment': sector_sentiments
        }

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        return macd

    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: int = 2) -> tuple:
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, lower

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

if __name__ == "__main__":
    # Clear cache to ensure fresh data
    data_dir = 'datacollection'
    os.makedirs(data_dir, exist_ok=True)
    for file in os.listdir(data_dir):
        if file.endswith('.json') or file in ['fred_data.csv', 'vix_data.csv']:
            os.remove(os.path.join(data_dir, file))
            print(f"Removed cache file: {file}")
    
    # Collect data
    collector = EnhancedDataCollector()
    data = collector.collect_all_data(start_date='2022-06-05', end_date='2024-12-31')
    
    # Save data for next step
    with open(os.path.join(data_dir, 'collected_data.pkl'), 'wb') as f:
        pickle.dump(data, f)
    print("Saved collected data to 'datacollection/collected_data.pkl'")