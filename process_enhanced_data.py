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
            
            # Add economic indicators if available
            if 'economic' in market_data:
                economic_data = market_data['economic']
                # Resample economic data to match stock data frequency
                for col in economic_data.columns:
                    if col != 'Date':
                        df[f'econ_{col}'] = economic_data[col].reindex(df.index, method='ffill')
            
            # Add sector-specific sentiment
            sector = df['Sector'].iloc[0]  # Get sector from the dataframe
            if 'sentiment' in market_data and 'market_sentiment' in market_data['sentiment']:
                sentiment = market_data['sentiment']['market_sentiment'].get(sector, 0)
                df['sentiment_score'] = sentiment
            
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
            if any(col.startswith('econ_') for col in df.columns):
                gdp_growth = df['econ_GDP_Growth'].iloc[-1] if 'econ_GDP_Growth' in df.columns else 0
                unemployment = df['econ_Unemployment'].iloc[-1] if 'econ_Unemployment' in df.columns else 0
                inflation = df['econ_Inflation'].iloc[-1] if 'econ_Inflation' in df.columns else 0
                yield_curve = df['econ_Yield_Curve'].iloc[-1] if 'econ_Yield_Curve' in df.columns else 0
                
                signals[ticker]['fundamental'] = {
                    'gdp_growth': 'positive' if gdp_growth > 0 else 'negative',
                    'unemployment': 'high' if unemployment > 6 else 'moderate' if unemployment > 4 else 'low',
                    'inflation': 'high' if inflation > 3 else 'moderate' if inflation > 2 else 'low',
                    'yield_curve': 'normal' if yield_curve > 0 else 'inverted'
                }
            
            # Sentiment Signals
            if 'sentiment_score' in df.columns:
                sentiment = df['sentiment_score'].iloc[-1]
                signals[ticker]['sentiment'] = {
                    'score': sentiment,
                    'indication': 'bullish' if sentiment > 0.2 else 'bearish' if sentiment < -0.2 else 'neutral'
                }
            
            # Relative Strength Signals
            if 'relative_strength' in df.columns:
                rel_strength = df['relative_strength'].iloc[-1]
                signals[ticker]['relative'] = {
                    'vs_sector': 'outperforming' if rel_strength > 1.05 else 
                                'underperforming' if rel_strength < 0.95 else 'neutral',
                    'strength': rel_strength
                }
            
            # Combined Signal
            technical_score = (
                (1 if signals[ticker]['technical']['trend']['short_term'] == 'bullish' else -1) +
                (1 if signals[ticker]['technical']['trend']['long_term'] == 'bullish' else -1) +
                (1 if signals[ticker]['technical']['momentum']['macd'] == 'bullish' else -1)
            )
            
            fundamental_score = sum([
                1 if 'fundamental' in signals[ticker] and signals[ticker]['fundamental']['gdp_growth'] == 'positive' else -1,
                1 if 'fundamental' in signals[ticker] and signals[ticker]['fundamental']['yield_curve'] == 'normal' else -1,
                -1 if 'fundamental' in signals[ticker] and signals[ticker]['fundamental']['inflation'] == 'high' else 0
            ])
            
            sentiment_score = 1 if 'sentiment' in signals[ticker] and signals[ticker]['sentiment']['indication'] == 'bullish' else \
                            -1 if 'sentiment' in signals[ticker] and signals[ticker]['sentiment']['indication'] == 'bearish' else 0
            
            relative_score = 1 if 'relative' in signals[ticker] and signals[ticker]['relative']['vs_sector'] == 'outperforming' else \
                           -1 if 'relative' in signals[ticker] and signals[ticker]['relative']['vs_sector'] == 'underperforming' else 0
            
            total_score = technical_score + fundamental_score + sentiment_score + relative_score
            
            signals[ticker]['combined'] = {
                'score': total_score,
                'recommendation': 'strong_buy' if total_score >= 3 else
                                'buy' if total_score > 0 else
                                'strong_sell' if total_score <= -3 else
                                'sell' if total_score < 0 else 'hold',
                'confidence': abs(total_score) / 8  # Normalize confidence score
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
        print(f"\n{ticker} ({signal.get('Type', 'Unknown')} - {signal.get('Sector', 'Unknown')}):")
        print(f"Technical Analysis: {signal['technical']}")
        if signal['fundamental']:
            print(f"Fundamental Analysis: {signal['fundamental']}")
        if signal['sentiment']:
            print(f"Sentiment Analysis: {signal['sentiment']}")
        if signal['relative']:
            print(f"Relative Strength: {signal['relative']}")
        print(f"Combined Recommendation: {signal['combined']['recommendation']} (Confidence: {signal['combined']['confidence']:.2f})")
    
    # Save signals to file with custom encoder
    with open('fundamental_data.json', 'w') as f:
        json.dump(signals, f, indent=4, cls=NumpyEncoder)
    print("\nSignals saved to fundamental_data.json")

if __name__ == "__main__":
    main()