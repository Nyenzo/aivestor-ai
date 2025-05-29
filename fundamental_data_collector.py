import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime, timedelta
import yfinance as yf
import ta
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class FundamentalDataCollector:
    def __init__(self):
        """Initialize the data collector with API keys and configurations"""
        self.fred = Fred(api_key=os.getenv('FRED_API_KEY'))
        self.fred_indicators = {
            'FEDFUNDS': 'Federal Funds Rate',
            'CPIAUCSL': 'Consumer Price Index',
            'UNRATE': 'Unemployment Rate',
            'GDP': 'Gross Domestic Product',
            'M2': 'M2 Money Supply',
            'T10Y2Y': '10-Year Treasury Constant Maturity Minus 2-Year',
            'VIXCLS': 'CBOE Volatility Index',
            'INDPRO': 'Industrial Production Index',
            'HOUST': 'Housing Starts',
            'RETAILSMNSA': 'Retail Sales'
        }
        
        self.technical_indicators = {
            'trend': [
                'sma', 'ema', 'macd', 'adx', 'cci', 'dpo', 'ichimoku', 'psar', 'trix'
            ],
            'momentum': [
                'rsi', 'stoch', 'stoch_rsi', 'williams_r', 'ultimate_oscillator'
            ],
            'volatility': [
                'bbands', 'keltner', 'donchian', 'atr'
            ],
            'volume': [
                'acc_dist_index', 'chaikin_money_flow', 'ease_of_movement', 'force_index', 'money_flow_index', 'volume_price_trend'
            ]
        }

    def get_fred_data(self, start_date='2018-01-01'):
        """Collect economic indicators from FRED"""
        print("Collecting FRED economic indicators...")
        fred_data = pd.DataFrame()
        
        for series_id, name in self.fred_indicators.items():
            try:
                series = self.fred.get_series(series_id, start_date)
                series = series.reindex(pd.date_range(start=start_date, end=datetime.now()))
                series = series.fillna(method='ffill')  # Forward fill missing values
                fred_data[name] = series
                print(f"Collected {name} data")
            except Exception as e:
                print(f"Error collecting {name}: {e}")
                
        return fred_data

    def calculate_technical_indicators(self, df):
        """Calculate comprehensive technical indicators"""
        print("Calculating technical indicators...")
        
        # Initialize an empty DataFrame for all indicators
        indicators_df = pd.DataFrame(index=df.index)
        
        # Trend Indicators
        if 'sma' in self.technical_indicators['trend']:
            for period in [20, 50, 200]:
                indicators_df[f'SMA_{period}'] = ta.trend.sma_indicator(df['Close'], period)
                indicators_df[f'SMA_{period}_Distance'] = (df['Close'] - indicators_df[f'SMA_{period}']) / df['Close']

        if 'ema' in self.technical_indicators['trend']:
            for period in [12, 26, 50]:
                indicators_df[f'EMA_{period}'] = ta.trend.ema_indicator(df['Close'], period)
                
        if 'macd' in self.technical_indicators['trend']:
            macd = ta.trend.MACD(df['Close'])
            indicators_df['MACD'] = macd.macd()
            indicators_df['MACD_Signal'] = macd.macd_signal()
            indicators_df['MACD_Hist'] = macd.macd_diff()
            
        if 'adx' in self.technical_indicators['trend']:
            indicators_df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
            
        # Momentum Indicators
        if 'rsi' in self.technical_indicators['momentum']:
            indicators_df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            
        if 'stoch' in self.technical_indicators['momentum']:
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            indicators_df['Stoch_K'] = stoch.stoch()
            indicators_df['Stoch_D'] = stoch.stoch_signal()
            
        # Volatility Indicators
        if 'bbands' in self.technical_indicators['volatility']:
            bb = ta.volatility.BollingerBands(df['Close'])
            indicators_df['BB_High'] = bb.bollinger_hband()
            indicators_df['BB_Mid'] = bb.bollinger_mavg()
            indicators_df['BB_Low'] = bb.bollinger_lband()
            indicators_df['BB_Width'] = (indicators_df['BB_High'] - indicators_df['BB_Low']) / indicators_df['BB_Mid']
            
        if 'atr' in self.technical_indicators['volatility']:
            indicators_df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
            
        # Volume Indicators
        if 'volume_price_trend' in self.technical_indicators['volume']:
            indicators_df['Volume_Price_Trend'] = ta.volume.volume_price_trend(df['Close'], df['Volume'])
            
        if 'money_flow_index' in self.technical_indicators['volume']:
            indicators_df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
            
        return indicators_df

    def get_stock_data(self, ticker, start_date='2018-01-01'):
        """Fetch extended historical stock data with technical indicators"""
        print(f"Collecting historical data for {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date)
            
            if df.empty:
                print(f"No data found for {ticker}")
                return None
                
            # Calculate returns and volatility
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Volatility'] = df['Returns'].rolling(window=21).std() * np.sqrt(252)  # Annualized volatility
            
            # Add technical indicators
            technical_indicators = self.calculate_technical_indicators(df)
            df = pd.concat([df, technical_indicators], axis=1)
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

    def collect_all_data(self, tickers, start_date='2018-01-01'):
        """Collect all fundamental and technical data"""
        # Get economic indicators
        fred_data = self.get_fred_data(start_date)
        
        # Get stock data for each ticker
        stock_data = {}
        for ticker in tickers:
            stock_data[ticker] = self.get_stock_data(ticker, start_date)
            
        return {
            'economic_indicators': fred_data,
            'stock_data': stock_data
        }

    def save_data(self, data, base_path='.'):
        """Save collected data to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save economic indicators
        data['economic_indicators'].to_csv(f'{base_path}/economic_indicators_{timestamp}.csv')
        
        # Save stock data
        for ticker, df in data['stock_data'].items():
            if df is not None:
                df.to_csv(f'{base_path}/stock_data_{ticker}_{timestamp}.csv')
                
        print(f"Data saved with timestamp {timestamp}")
        return timestamp

def main():
    # Initialize collector
    collector = FundamentalDataCollector()
    
    # Define sector ETFs
    sector_etfs = {
        'Technology': 'XLK',
        'Healthcare': 'XLV',
        'Financials': 'XLF',
        'Consumer Discretionary': 'XLY',
        'Consumer Staples': 'XLP',
        'Energy': 'XLE',
        'Industrials': 'XLI',
        'Utilities': 'XLU'
    }
    
    # Collect data from 2018 (5 years of historical data)
    data = collector.collect_all_data(sector_etfs.values(), start_date='2018-01-01')
    
    # Save the data
    timestamp = collector.save_data(data)
    print("Data collection completed successfully!")

if __name__ == "__main__":
    main() 