import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
from datetime import datetime, timedelta
import joblib
from transformers import pipeline
import torch
import ta
import warnings
warnings.filterwarnings('ignore')

class AdvancedStockPredictor:
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
        
        self.sentiment_analyzer = pipeline('sentiment-analysis', 
                                        model='distilbert-base-uncased-finetuned-sst-2-english')
        
        self.short_term_model = None
        self.long_term_model = None
        self.scaler = StandardScaler()
        
    def fetch_stock_data(self, ticker, period='2y'):
        """Fetch historical stock data and calculate technical indicators"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty:
                return None
                
            # Calculate technical indicators
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            df['MACD'] = ta.trend.MACD(df['Close']).macd()
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband(), \
                                                             ta.volatility.BollingerBands(df['Close']).bollinger_mavg(), \
                                                             ta.volatility.BollingerBands(df['Close']).bollinger_lband()
            
            # Calculate returns
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            
            # Add moving averages
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
            
    def prepare_features(self, df, sentiment_score, sector_sentiment):
        """Prepare features for prediction"""
        features = pd.DataFrame()
        
        # Technical indicators
        features['RSI'] = df['RSI'].fillna(50)
        features['MACD'] = df['MACD'].fillna(0)
        features['BB_Position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        features['Volatility'] = df['Volatility'].fillna(df['Volatility'].mean())
        
        # Trend indicators
        features['SMA_50_Position'] = (df['Close'] - df['SMA_50']) / df['Close']
        features['SMA_200_Position'] = (df['Close'] - df['SMA_200']) / df['Close']
        
        # Volume analysis
        features['Volume_Change'] = df['Volume'].pct_change()
        
        # Sentiment features
        features['News_Sentiment'] = sentiment_score
        features['Sector_Sentiment'] = sector_sentiment
        
        # Fill any remaining NaN values
        features = features.fillna(0)
        
        return features
        
    def train_models(self, sector_data, sentiment_results):
        """Train short-term and long-term prediction models"""
        X = []
        y_short = []
        y_long = []
        
        for sector, ticker in self.sector_etfs.items():
            df = self.fetch_stock_data(ticker)
            if df is None:
                continue
                
            # Get sentiment scores
            sector_sentiment = sentiment_results.get(sector, {'sentiment': 0.0})['sentiment']
            
            # Prepare features
            features = self.prepare_features(df, 
                                          sentiment_score=sector_sentiment,
                                          sector_sentiment=sector_sentiment)
            
            # Create labels
            short_term_returns = df['Close'].pct_change(periods=63)  # ~3 months
            long_term_returns = df['Close'].pct_change(periods=252)  # ~1 year
            
            # Add to training data
            X.extend(features.values)
            y_short.extend((short_term_returns > 0).astype(int))
            y_long.extend((long_term_returns > 0).astype(int))
        
        # Convert to numpy arrays
        X = np.array(X)
        y_short = np.array(y_short)
        y_long = np.array(y_long)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        self.short_term_model = RandomForestClassifier(n_estimators=200, 
                                                      max_depth=10,
                                                      random_state=42)
        self.long_term_model = RandomForestClassifier(n_estimators=200,
                                                     max_depth=10,
                                                     random_state=42)
        
        self.short_term_model.fit(X_scaled, y_short)
        self.long_term_model.fit(X_scaled, y_long)
        
    def predict_sector(self, sector, ticker, sentiment_score):
        """Make predictions for a specific sector"""
        df = self.fetch_stock_data(ticker, period='6mo')
        if df is None:
            return None
            
        # Prepare features
        features = self.prepare_features(df.iloc[-1:], 
                                       sentiment_score=sentiment_score,
                                       sector_sentiment=sentiment_score)
        
        # Scale features
        X_scaled = self.scaler.transform(features)
        
        # Make predictions
        short_term_prob = self.short_term_model.predict_proba(X_scaled)[0]
        long_term_prob = self.long_term_model.predict_proba(X_scaled)[0]
        
        return {
            'sector': sector,
            'short_term': {
                'prediction': 'Bullish' if short_term_prob[1] > 0.6 else 'Bearish' if short_term_prob[1] < 0.4 else 'Neutral',
                'confidence': float(max(short_term_prob)),
                'probability': float(short_term_prob[1])
            },
            'long_term': {
                'prediction': 'Bullish' if long_term_prob[1] > 0.6 else 'Bearish' if long_term_prob[1] < 0.4 else 'Neutral',
                'confidence': float(max(long_term_prob)),
                'probability': float(long_term_prob[1])
            },
            'technical_indicators': {
                'RSI': float(features['RSI'].iloc[-1]),
                'MACD': float(features['MACD'].iloc[-1]),
                'Volatility': float(features['Volatility'].iloc[-1])
            }
        }
        
    def generate_portfolio_recommendation(self, predictions, risk_tolerance='moderate'):
        """Generate portfolio recommendations based on predictions and risk tolerance"""
        risk_weights = {
            'conservative': {'short_term': 0.2, 'long_term': 0.8},
            'moderate': {'short_term': 0.4, 'long_term': 0.6},
            'aggressive': {'short_term': 0.6, 'long_term': 0.4}
        }
        
        weights = risk_weights.get(risk_tolerance, risk_weights['moderate'])
        
        # Calculate sector scores
        sector_scores = {}
        for sector, pred in predictions.items():
            short_score = (1 if pred['short_term']['prediction'] == 'Bullish' else 
                         -1 if pred['short_term']['prediction'] == 'Bearish' else 0)
            long_score = (1 if pred['long_term']['prediction'] == 'Bullish' else 
                         -1 if pred['long_term']['prediction'] == 'Bearish' else 0)
            
            # Weight the scores based on risk tolerance
            weighted_score = (short_score * weights['short_term'] * pred['short_term']['confidence'] +
                            long_score * weights['long_term'] * pred['long_term']['confidence'])
            
            sector_scores[sector] = weighted_score
            
        # Sort sectors by score
        sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Generate recommendations
        recommendations = {
            'top_sectors': [],
            'neutral_sectors': [],
            'avoid_sectors': []
        }
        
        for sector, score in sorted_sectors:
            if score > 0.3:
                recommendations['top_sectors'].append({
                    'sector': sector,
                    'score': score,
                    'etf': self.sector_etfs[sector],
                    'confidence': predictions[sector]['long_term']['confidence']
                })
            elif score < -0.3:
                recommendations['avoid_sectors'].append({
                    'sector': sector,
                    'score': score,
                    'etf': self.sector_etfs[sector],
                    'confidence': predictions[sector]['long_term']['confidence']
                })
            else:
                recommendations['neutral_sectors'].append({
                    'sector': sector,
                    'score': score,
                    'etf': self.sector_etfs[sector],
                    'confidence': predictions[sector]['long_term']['confidence']
                })
                
        return recommendations

def save_models(predictor, base_path='.'):
    """Save trained models and scaler"""
    joblib.dump(predictor.short_term_model, f'{base_path}/stock_short_term_model.pkl')
    joblib.dump(predictor.long_term_model, f'{base_path}/stock_long_term_model.pkl')
    joblib.dump(predictor.scaler, f'{base_path}/stock_scaler.pkl')

def load_models(predictor, base_path='.'):
    """Load trained models and scaler"""
    predictor.short_term_model = joblib.load(f'{base_path}/stock_short_term_model.pkl')
    predictor.long_term_model = joblib.load(f'{base_path}/stock_long_term_model.pkl')
    predictor.scaler = joblib.load(f'{base_path}/stock_scaler.pkl') 