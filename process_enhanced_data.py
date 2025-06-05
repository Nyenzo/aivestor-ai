# Process collected data into features and trading signals for all sectors
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.preprocessing import StandardScaler

class EnhancedDataProcessor:
    def __init__(self):
        # Initialize feature processing components
        self.scaler = StandardScaler()

    def prepare_features(self, collected_data: Dict) -> Dict[str, pd.DataFrame]:
        # Process collected data into feature set for model training
        stock_data = collected_data['stock_data']
        economic_data = collected_data.get('economic_data', pd.DataFrame())
        market_sentiment = collected_data.get('market_sentiment', {})
        
        processed_features = {}
        for ticker, df in stock_data.items():
            df = df.copy()
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Price-based features
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=21).std()
            
            # Volume features
            df['Volume_Change'] = df['Volume'].pct_change()
            
            # Technical indicators (already included: RSI, MACD, BB, ATR)
            
            # Economic indicators
            if not economic_data.empty:
                economic_data['Date'] = pd.to_datetime(economic_data['Date'])
                economic_data.set_index('Date', inplace=True)
                df = df.join(economic_data, how='left')
                df[['GDP', 'Inflation', 'Unemployment', 'FedFundsRate']] = df[['GDP', 'Inflation', 'Unemployment', 'FedFundsRate']].fillna(method='ffill')
            
            # Sentiment features
            from advanced_stock_predictor import AdvancedStockPredictor
            predictor = AdvancedStockPredictor()
            sector = None
            for sec, tickers in predictor.sector_tickers.items():
                if ticker in tickers or ticker == predictor.sector_etfs.get(sec):
                    sector = sec
                    break
            if sector and sector in market_sentiment:
                sentiment_scores = [s['score'] if s['sentiment'] == 'POSITIVE' else -s['score'] 
                                  for s in market_sentiment[sector]]
                df['Sentiment'] = np.mean(sentiment_scores) if sentiment_scores else 0.0
            else:
                df['Sentiment'] = 0.0
            
            df = df.dropna()
            processed_features[ticker] = df
        return processed_features

    def generate_signals(self, processed_features: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        # Generate trading signals based on features
        signals = {}
        for ticker, df in processed_features.items():
            signals[ticker] = {}
            
            # Technical signals
            signals[ticker]['RSI'] = 'buy' if df['RSI'].iloc[-1] < 30 else 'sell' if df['RSI'].iloc[-1] > 70 else 'hold'
            signals[ticker]['MACD'] = 'buy' if df['MACD'].iloc[-1] > 0 else 'sell'
            
            # Fundamental signals (simplified)
            signals[ticker]['Sentiment'] = 'buy' if df['Sentiment'].iloc[-1] > 0.5 else 'sell' if df['Sentiment'].iloc[-1] < -0.5 else 'hold'
            
            # Combined signal
            buy_signals = sum(1 for s in signals[ticker].values() if s == 'buy')
            sell_signals = sum(1 for s in signals[ticker].values() if s == 'sell')
            signals[ticker]['Combined'] = 'buy' if buy_signals >= 2 else 'sell' if sell_signals >= 2 else 'hold'
        
        return signals