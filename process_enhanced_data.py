import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from sklearn.preprocessing import StandardScaler
import ta

class EnhancedDataProcessor:
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
        self.scaler = StandardScaler()
        
    def load_market_data(self):
        """Load the collected market data"""
        market_data = {}
        
        # Load market sentiment
        try:
            with open('market_data.json', 'r') as f:
                market_data['sentiment'] = json.load(f)
        except FileNotFoundError:
            print("Market sentiment data not found")
            market_data['sentiment'] = {}
            
        # Load stock data for both ETFs and individual stocks
        stock_data = {}
        
        # Load ETF data
        for sector, ticker in self.sector_etfs.items():
            files = [f for f in os.listdir() if f.startswith(f'stock_data_{ticker}_')]
            if files:
                latest_file = max(files)
                stock_data[ticker] = pd.read_csv(latest_file)
                stock_data[ticker]['Date'] = pd.to_datetime(stock_data[ticker]['Date'], utc=True)
                stock_data[ticker].set_index('Date', inplace=True)
                stock_data[ticker]['Type'] = 'ETF'
                stock_data[ticker]['Sector'] = sector
        
        # Load individual stock data
        for sector, tickers in self.sector_tickers.items():
            for ticker in tickers:
                files = [f for f in os.listdir() if f.startswith(f'stock_data_{ticker}_')]
                if files:
                    latest_file = max(files)
                    stock_data[ticker] = pd.read_csv(latest_file)
                    stock_data[ticker]['Date'] = pd.to_datetime(stock_data[ticker]['Date'], utc=True)
                    stock_data[ticker].set_index('Date', inplace=True)
                    stock_data[ticker]['Type'] = 'Stock'
                    stock_data[ticker]['Sector'] = sector
                    
        market_data['stock_data'] = stock_data
        
        # Load economic indicators with enhanced handling
        econ_files = [f for f in os.listdir() if f.startswith('economic_indicators_')]
        if econ_files:
            latest_econ = max(econ_files)
            economic_data = pd.read_csv(latest_econ)
            if 'Unnamed: 0' in economic_data.columns:  # Handle index column if present
                economic_data.set_index('Unnamed: 0', inplace=True)
                economic_data.index.name = 'Date'
            economic_data.index = pd.to_datetime(economic_data.index, utc=True)
            
            # Calculate additional derived indicators
            if 'Fed_Funds_Rate' in economic_data.columns and '10Y_Treasury' in economic_data.columns:
                economic_data['Yield_Curve'] = economic_data['10Y_Treasury'] - economic_data['Fed_Funds_Rate']
            
            if 'GDP' in economic_data.columns:
                economic_data['GDP_Growth'] = economic_data['GDP'].pct_change()
            
            if 'Nonfarm_Payrolls' in economic_data.columns:
                economic_data['Payrolls_Change'] = economic_data['Nonfarm_Payrolls'].pct_change()
            
            market_data['economic'] = economic_data
            
        return market_data
        
    def calculate_technical_indicators(self, df):
        """Calculate or update technical indicators for a dataframe"""
        # Trend Indicators
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        
        # Momentum Indicators
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # Volatility Indicators
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['Close']
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        # Volume Indicators
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        
        return df
        
    def prepare_features(self, market_data, lookback_periods=[5, 10, 20, 50]):
        """Prepare features for model training with enhanced feature set"""
        features = {}
        
        for ticker, df in market_data['stock_data'].items():
            # Calculate returns for multiple timeframes
            for period in lookback_periods:
                df[f'return_{period}d'] = df['Close'].pct_change(period)
                df[f'volume_{period}d_avg'] = df['Volume'].rolling(period).mean()
                df[f'volatility_{period}d'] = df['Close'].pct_change().rolling(period).std()
            
            # Add technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Add fundamental data if available
            try:
                fundamental_data = self.load_fundamental_data(ticker)
                if fundamental_data:
                    # Valuation Metrics
                    df['PE_Ratio'] = fundamental_data.get('PE_Ratio')
                    df['PB_Ratio'] = fundamental_data.get('PB_Ratio')
                    df['PS_Ratio'] = fundamental_data.get('PS_Ratio')
                    df['PEG_Ratio'] = fundamental_data.get('PEG_Ratio')
                    
                    # Financial Health
                    df['Debt_to_Equity'] = fundamental_data.get('Debt_to_Equity')
                    df['Current_Ratio'] = fundamental_data.get('Current_Ratio')
                    df['Quick_Ratio'] = fundamental_data.get('Quick_Ratio')
                    
                    # Profitability
                    df['ROE'] = fundamental_data.get('ROE')
                    df['ROA'] = fundamental_data.get('ROA')
                    df['Profit_Margin'] = fundamental_data.get('Profit_Margin')
                    df['Operating_Margin'] = fundamental_data.get('Operating_Margin')
                    
                    # Growth & Performance
                    df['EPS'] = fundamental_data.get('EPS')
                    df['EPS_Growth'] = fundamental_data.get('EPS_Growth')
                    df['Revenue_Growth'] = fundamental_data.get('Revenue_Growth')
                    
                    # Dividend Information
                    df['Dividend_Yield'] = fundamental_data.get('Dividend_Yield')
                    df['Dividend_Rate'] = fundamental_data.get('Dividend_Rate')
                    df['Payout_Ratio'] = fundamental_data.get('Payout_Ratio')
                    
                    # Cash Flow Analysis
                    df['Free_Cash_Flow'] = fundamental_data.get('Free_Cash_Flow')
                    df['Operating_Cash_Flow'] = fundamental_data.get('Operating_Cash_Flow')
                    
                    # Market Position
                    df['Market_Cap'] = fundamental_data.get('Market_Cap')
                    df['Enterprise_Value'] = fundamental_data.get('Enterprise_Value')
                    df['Beta'] = fundamental_data.get('Beta')
            except Exception as e:
                print(f"Error loading fundamental data for {ticker}: {e}")
            
            # Add economic indicators if available
            if 'economic' in market_data:
                economic_data = market_data['economic']
                # Resample economic data to match stock data frequency
                for col in economic_data.columns:
                    if col != 'Date':
                        df[f'econ_{col}'] = economic_data[col].reindex(df.index, method='ffill')
            
            # Add sector-specific sentiment and industry trends
            sector = df['Sector'].iloc[0]
            try:
                industry_trends = self.load_industry_trends(sector)
                if industry_trends:
                    df['sector_performance'] = industry_trends.get('Sector_Performance', {}).get('YTD_Return')
                    df['sector_volatility'] = industry_trends.get('Sector_Performance', {}).get('Volatility')
                    df['sector_volume_trend'] = industry_trends.get('Sector_Performance', {}).get('Volume_Trend')
            except Exception as e:
                print(f"Error loading industry trends for {sector}: {e}")
            
            # Add relative strength vs sector
            sector_etf = self.sector_etfs[sector]
            if sector_etf in market_data['stock_data']:
                etf_data = market_data['stock_data'][sector_etf]
                df['relative_strength'] = df['Close'] / df['Close'].iloc[0] / (etf_data['Close'] / etf_data['Close'].iloc[0])
            
            # Handle missing values
            df.ffill(inplace=True)  # Forward fill
            df.bfill(inplace=True)  # Backward fill for any remaining NAs at the start
            df.fillna(0, inplace=True)  # For any still remaining NAs
            
            features[ticker] = df
            
        return features
        
    def load_fundamental_data(self, ticker):
        """Load fundamental data for a specific ticker"""
        try:
            with open('fundamental_data.json', 'r') as f:
                fundamental_data = json.load(f)
                
            # Find the sector for this ticker
            sector = next((sector for sector, tickers in self.sector_tickers.items() 
                         if ticker in tickers), None)
            
            if sector and sector in fundamental_data:
                return fundamental_data[sector].get(ticker)
        except Exception as e:
            print(f"Error loading fundamental data: {e}")
        return None
        
    def load_industry_trends(self, sector):
        """Load industry trends for a specific sector"""
        try:
            with open('industry_trends.json', 'r') as f:
                trends_data = json.load(f)
                return trends_data.get(sector)
        except Exception as e:
            print(f"Error loading industry trends: {e}")
        return None
        
    def generate_signals(self, features):
        """Generate trading signals with enhanced analysis"""
        signals = {}
        
        for ticker, df in features.items():
            signals[ticker] = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'technical': {},
                'fundamental': {},
                'sentiment': {},
                'relative': {},
                'combined': {}
            }
            
            # Technical Signals
            signals[ticker]['technical'] = {
                'trend': {
                    'short_term': 'bullish' if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] else 'bearish',
                    'long_term': 'bullish' if df['Close'].iloc[-1] > df['SMA_200'].iloc[-1] else 'bearish'
                },
                'momentum': {
                    'rsi': 'overbought' if df['RSI'].iloc[-1] > 70 else 'oversold' if df['RSI'].iloc[-1] < 30 else 'neutral',
                    'macd': 'bullish' if df['MACD_Hist'].iloc[-1] > 0 else 'bearish'
                },
                'volatility': {
                    'level': 'high' if df['BB_width'].iloc[-1] > df['BB_width'].mean() else 'low',
                    'bollinger': 'overbought' if df['Close'].iloc[-1] > df['BB_upper'].iloc[-1] else 
                               'oversold' if df['Close'].iloc[-1] < df['BB_lower'].iloc[-1] else 'neutral'
                }
            }
            
            # Fundamental Signals
            if 'PE_Ratio' in df.columns:
                signals[ticker]['fundamental'] = {
                    'valuation': {
                        'pe_ratio': 'high' if df['PE_Ratio'].iloc[-1] > 25 else 'low' if df['PE_Ratio'].iloc[-1] < 15 else 'moderate',
                        'pb_ratio': 'high' if df['PB_Ratio'].iloc[-1] > 3 else 'low' if df['PB_Ratio'].iloc[-1] < 1 else 'moderate',
                        'dividend_yield': 'high' if df['Dividend_Yield'].iloc[-1] > 0.04 else 'low' if df['Dividend_Yield'].iloc[-1] < 0.02 else 'moderate'
                    },
                    'financial_health': {
                        'debt_equity': 'high' if df['Debt_to_Equity'].iloc[-1] > 2 else 'low' if df['Debt_to_Equity'].iloc[-1] < 1 else 'moderate',
                        'current_ratio': 'strong' if df['Current_Ratio'].iloc[-1] > 2 else 'weak' if df['Current_Ratio'].iloc[-1] < 1 else 'moderate'
                    },
                    'growth': {
                        'revenue': 'strong' if df['Revenue_Growth'].iloc[-1] > 0.1 else 'weak' if df['Revenue_Growth'].iloc[-1] < 0 else 'moderate',
                        'eps': 'strong' if df['EPS_Growth'].iloc[-1] > 0.1 else 'weak' if df['EPS_Growth'].iloc[-1] < 0 else 'moderate'
                    },
                    'profitability': {
                        'roe': 'strong' if df['ROE'].iloc[-1] > 0.15 else 'weak' if df['ROE'].iloc[-1] < 0.1 else 'moderate',
                        'profit_margin': 'strong' if df['Profit_Margin'].iloc[-1] > 0.2 else 'weak' if df['Profit_Margin'].iloc[-1] < 0.1 else 'moderate'
                    }
                }
            
            # Combine all signals for final recommendation
            bullish_signals = 0
            total_signals = 0
            
            # Count technical signals
            for category in signals[ticker]['technical'].values():
                if isinstance(category, dict):
                    for signal in category.values():
                        total_signals += 1
                        if signal in ['bullish', 'strong', 'oversold']:
                            bullish_signals += 1
                        elif signal in ['bearish', 'weak', 'overbought']:
                            bullish_signals -= 1
            
            # Count fundamental signals
            if 'fundamental' in signals[ticker]:
                for category in signals[ticker]['fundamental'].values():
                    if isinstance(category, dict):
                        for signal in category.values():
                            total_signals += 1
                            if signal in ['strong', 'low']:  # low for valuation metrics is good
                                bullish_signals += 1
                            elif signal in ['weak', 'high']:  # high for valuation metrics is bad
                                bullish_signals -= 1
            
            # Generate final recommendation
            sentiment_score = bullish_signals / total_signals if total_signals > 0 else 0
            signals[ticker]['combined'] = {
                'sentiment_score': sentiment_score,
                'recommendation': 'strong_buy' if sentiment_score > 0.5 else
                                'buy' if sentiment_score > 0.2 else
                                'strong_sell' if sentiment_score < -0.5 else
                                'sell' if sentiment_score < -0.2 else
                                'hold'
            }
            
        return signals

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
            
def main():
    processor = EnhancedDataProcessor()
    print("Loading market data...")
    market_data = processor.load_market_data()
    
    print("\nPreparing features...")
    features = processor.prepare_features(market_data)
    
    print("\nGenerating signals...")
    signals = processor.generate_signals(features)
    
    print("\nSignals Summary:")
    for ticker, signal in signals.items():
        print(f"\n{ticker}:")
        print(f"Technical Analysis: {signal['technical']}")
        if 'fundamental' in signal and signal['fundamental']:
            print(f"Fundamental Analysis: {signal['fundamental']}")
        if 'sentiment' in signal and signal['sentiment']:
            print(f"Sentiment Analysis: {signal['sentiment']}")
        if 'relative' in signal and signal['relative']:
            print(f"Relative Strength: {signal['relative']}")
        print(f"Combined Recommendation: {signal['combined']['recommendation']} (Score: {signal['combined']['sentiment_score']:.2f})")
    
    # Save signals to file with custom encoder
    try:
        with open('signals_output.json', 'w') as f:
            json.dump(signals, f, indent=4, cls=NumpyEncoder)
        print("\nSignals saved to signals_output.json")
    except Exception as e:
        print(f"\nError saving signals: {e}")
        
if __name__ == "__main__":
    main() 