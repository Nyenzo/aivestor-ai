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
from fundamental_data_collector import FundamentalDataCollector
from typing import Dict, Any
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
        
        self.short_term_model = RandomForestClassifier(n_estimators=200, 
                                                     max_depth=10,
                                                     random_state=42)
        self.long_term_model = RandomForestClassifier(n_estimators=200,
                                                    max_depth=10,
                                                    random_state=42)
        self.scaler = StandardScaler()
        self.data_collector = FundamentalDataCollector()
        
    def prepare_features(self, stock_data: Dict[str, pd.DataFrame], signals: Dict[str, Any]) -> pd.DataFrame:
        """Prepare features for model training"""
        features = []
        
        for ticker, data in stock_data.items():
            if ticker not in signals:
                continue
                
            ticker_features = data.copy()
            
            # Convert datetime index to timezone-naive
            if isinstance(ticker_features.index, pd.DatetimeIndex):
                if ticker_features.index.tz is not None:
                    ticker_features.index = ticker_features.index.tz_localize(None)
            
            # Add signal features
            signal = signals[ticker]
            ticker_features['sentiment_score'] = signal['Sentiment Analysis']['score']
            ticker_features['relative_strength'] = signal['Relative Strength']['strength']
            
            # Add technical signals
            tech_analysis = signal['Technical Analysis']
            ticker_features['trend_short'] = 1 if tech_analysis['trend']['short_term'] == 'bullish' else -1
            ticker_features['trend_long'] = 1 if tech_analysis['trend']['long_term'] == 'bullish' else -1
            ticker_features['momentum_rsi'] = 1 if tech_analysis['momentum']['rsi'] == 'overbought' else (-1 if tech_analysis['momentum']['rsi'] == 'oversold' else 0)
            ticker_features['momentum_macd'] = 1 if tech_analysis['momentum']['macd'] == 'bullish' else -1
            
            # Add fundamental signals
            fund_analysis = signal['Fundamental Analysis']
            ticker_features['gdp_growth'] = 1 if fund_analysis['gdp_growth'] == 'positive' else -1
            ticker_features['unemployment'] = 1 if fund_analysis['unemployment'] == 'low' else -1
            ticker_features['inflation'] = 1 if fund_analysis['inflation'] == 'low' else -1
            ticker_features['yield_curve'] = 1 if fund_analysis['yield_curve'] == 'normal' else -1
            
            features.append(ticker_features)
            
        return pd.concat(features, axis=0)
        
    def train_models(self, sector_data=None, sentiment_results=None, start_date='2018-01-01'):
        """Train short-term and long-term prediction models with enhanced data"""
        print("Collecting enhanced historical and fundamental data...")
        
        # Collect enhanced data
        collected_data = self.data_collector.collect_all_data(
            self.sector_etfs.values(), 
            start_date=start_date
        )
        
        X_all = []
        y_short_all = []
        y_long_all = []
        
        for sector, ticker in self.sector_etfs.items():
            stock_data = collected_data['stock_data'].get(ticker)
            if stock_data is None:
                continue
                
            # Get sentiment scores
            sector_sentiment = sentiment_results.get(sector, {'sentiment': 0.0})['sentiment']
            
            # Prepare features with fundamental data
            features = self.prepare_features(
                stock_data=stock_data,
                economic_data=collected_data['economic_indicators'],
                sentiment_score=sector_sentiment
            )
            
            # Create labels
            short_term_returns = stock_data['Close'].pct_change(periods=63)  # ~3 months
            long_term_returns = stock_data['Close'].pct_change(periods=252)  # ~1 year
            
            # Remove rows with NaN values
            valid_idx = ~(features.isna().any(axis=1) | short_term_returns.isna() | long_term_returns.isna())
            
            if valid_idx.sum() > 0:
                X_all.extend(features[valid_idx].values)
                y_short_all.extend((short_term_returns[valid_idx] > 0).astype(int))
                y_long_all.extend((long_term_returns[valid_idx] > 0).astype(int))
        
        # Convert to numpy arrays
        X_all = np.array(X_all)
        y_short_all = np.array(y_short_all)
        y_long_all = np.array(y_long_all)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_all)
        
        # Train models
        print("Training models with enhanced features...")
        self.short_term_model.fit(X_scaled, y_short_all)
        self.long_term_model.fit(X_scaled, y_long_all)
        
        print("Model training completed!")
        
    def predict_sector(self, sector, ticker, sentiment_score):
        """Make predictions using enhanced feature set"""
        # Collect recent data
        collected_data = self.data_collector.collect_all_data(
            [ticker],
            start_date=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        )
        
        stock_data = collected_data['stock_data'].get(ticker)
        if stock_data is None:
            return None
            
        # Prepare features
        features = self.prepare_features(
            stock_data=stock_data.iloc[-1:],
            economic_data=collected_data['economic_indicators'],
            sentiment_score=sentiment_score
        )
        
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
                'RSI': float(features['RSI'].iloc[-1]) if 'RSI' in features else None,
                'MACD': float(features['MACD'].iloc[-1]) if 'MACD' in features else None,
                'Volatility': float(features['Volatility'].iloc[-1]) if 'Volatility' in features else None
            },
            'economic_indicators': {
                indicator: float(features[name].iloc[-1]) if name in features else None
                for indicator, name in self.data_collector.fred_indicators.items()
            }
        }
        
    def generate_portfolio_recommendation(self, predictions, risk_tolerance='moderate'):
        """Generate portfolio recommendations based on enhanced predictions"""
        risk_weights = {
            'conservative': {'short_term': 0.2, 'long_term': 0.8},
            'moderate': {'short_term': 0.4, 'long_term': 0.6},
            'aggressive': {'short_term': 0.6, 'long_term': 0.4}
        }
        
        weights = risk_weights.get(risk_tolerance, risk_weights['moderate'])
        
        # Calculate sector scores with enhanced metrics
        sector_scores = {}
        for sector, pred in predictions.items():
            # Calculate technical score
            tech_indicators = pred['technical_indicators']
            tech_score = 0
            if tech_indicators['RSI'] is not None:
                tech_score += 1 if 40 <= tech_indicators['RSI'] <= 60 else -1
            if tech_indicators['MACD'] is not None:
                tech_score += 1 if tech_indicators['MACD'] > 0 else -1
                
            # Calculate prediction scores
            short_score = (1 if pred['short_term']['prediction'] == 'Bullish' else 
                         -1 if pred['short_term']['prediction'] == 'Bearish' else 0)
            long_score = (1 if pred['long_term']['prediction'] == 'Bullish' else 
                         -1 if pred['long_term']['prediction'] == 'Bearish' else 0)
            
            # Combine scores with risk weights
            weighted_score = (
                short_score * weights['short_term'] * pred['short_term']['confidence'] +
                long_score * weights['long_term'] * pred['long_term']['confidence'] +
                tech_score * 0.2  # Technical analysis weight
            )
            
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
                    'confidence': predictions[sector]['long_term']['confidence'],
                    'technical_indicators': predictions[sector]['technical_indicators'],
                    'economic_indicators': predictions[sector].get('economic_indicators', {})
                })
            elif score < -0.3:
                recommendations['avoid_sectors'].append({
                    'sector': sector,
                    'score': score,
                    'etf': self.sector_etfs[sector],
                    'confidence': predictions[sector]['long_term']['confidence'],
                    'technical_indicators': predictions[sector]['technical_indicators'],
                    'economic_indicators': predictions[sector].get('economic_indicators', {})
                })
            else:
                recommendations['neutral_sectors'].append({
                    'sector': sector,
                    'score': score,
                    'etf': self.sector_etfs[sector],
                    'confidence': predictions[sector]['long_term']['confidence'],
                    'technical_indicators': predictions[sector]['technical_indicators'],
                    'economic_indicators': predictions[sector].get('economic_indicators', {})
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