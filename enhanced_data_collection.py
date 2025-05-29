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

# Load environment variables
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

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
        
        self.economic_indicators = {
            'GDP': 'GDP',
            'Inflation': 'CPIAUCSL',
            'Unemployment': 'UNRATE',
            'Interest Rate': 'DFF',
            'Consumer Sentiment': 'UMCSENT'
        }
        
        if FRED_API_KEY:
            self.fred = Fred(api_key=FRED_API_KEY)
        
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
        for indicator_name, series_id in self.economic_indicators.items():
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
            
    def collect_all_data(self):
        """Collect all enhanced data and save to files"""
        # Collect market data
        market_data = self.fetch_historical_market_data()
        self.save_data(market_data, 'market_data.json')
        
        # Collect economic indicators
        economic_data = self.fetch_economic_indicators()
        self.save_data(economic_data, 'economic_data.json')
        
        # Collect sector fundamentals
        fundamental_data = self.fetch_sector_fundamentals()
        self.save_data(fundamental_data, 'fundamental_data.json')
        
if __name__ == "__main__":
    collector = EnhancedDataCollector()
    collector.collect_all_data() 