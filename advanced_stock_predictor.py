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
from typing import Dict, Any, List, Union
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
        
        # Add individual stock tickers for each sector
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
        self.feature_importance = {}
        
    def get_all_tickers(self) -> List[str]:
        """Get all tickers (both ETFs and individual stocks)"""
        all_tickers = list(self.sector_etfs.values())
        for sector_stocks in self.sector_tickers.values():
            all_tickers.extend(sector_stocks)
        return all_tickers
        
    def prepare_features(self, stock_data: pd.DataFrame, economic_data: pd.DataFrame = None, 
                        sentiment_score: float = 0.0) -> pd.DataFrame:
        """Prepare features for model training with enhanced feature set"""
        features = pd.DataFrame()
        
        # Price-based features
        features['return_1d'] = stock_data['Close'].pct_change()
        features['return_5d'] = stock_data['Close'].pct_change(5)
        features['return_20d'] = stock_data['Close'].pct_change(20)
        features['return_60d'] = stock_data['Close'].pct_change(60)
        
        # Volume features
        features['volume_ratio'] = stock_data['Volume'] / stock_data['Volume'].rolling(20).mean()
        features['volume_trend'] = stock_data['Volume'].pct_change(5)
        
        # Volatility features
        features['volatility_5d'] = features['return_1d'].rolling(5).std()
        features['volatility_20d'] = features['return_1d'].rolling(20).std()
        
        # Technical indicators
        features['RSI'] = ta.momentum.RSIIndicator(stock_data['Close']).rsi()
        macd = ta.trend.MACD(stock_data['Close'])
        features['MACD'] = macd.macd_diff()
        features['MACD_Signal'] = macd.macd_signal()
        
        bb = ta.volatility.BollingerBands(stock_data['Close'])
        features['BB_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / stock_data['Close']
        
        # Add economic indicators if available
        if economic_data is not None:
            for col in economic_data.columns:
                features[f'econ_{col}'] = economic_data[col].reindex(stock_data.index).ffill()
        
        # Add sentiment score
        features['sentiment'] = sentiment_score
        
        # Handle missing values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
        
    def train_models(self, start_date: str = '2018-01-01'):
        """Train short-term and long-term prediction models with enhanced data"""
        print("Collecting enhanced historical and fundamental data...")
        
        # Get all tickers
        all_tickers = self.get_all_tickers()
        
        # Collect enhanced data
        collected_data = self.data_collector.collect_all_data(
            all_tickers, 
            start_date=start_date
        )
        
        X_all = []
        y_short_all = []
        y_long_all = []
        
        # Process ETFs
        for sector, etf in self.sector_etfs.items():
            self._process_ticker_data(collected_data, sector, etf, X_all, y_short_all, y_long_all, is_etf=True)
        
        # Process individual stocks
        for sector, tickers in self.sector_tickers.items():
            for ticker in tickers:
                self._process_ticker_data(collected_data, sector, ticker, X_all, y_short_all, y_long_all, is_etf=False)
        
        # Convert to numpy arrays and train models
        if X_all and y_short_all and y_long_all:
            self._train_final_models(X_all, y_short_all, y_long_all)
        
    def _process_ticker_data(self, collected_data: Dict, sector: str, ticker: str, 
                           X_all: List, y_short_all: List, y_long_all: List, is_etf: bool):
        """Process data for a single ticker"""
        stock_data = collected_data['stock_data'].get(ticker)
        if stock_data is None:
            return
            
            # Prepare features
        features = self.prepare_features(
            stock_data=stock_data,
            economic_data=collected_data.get('economic_indicators'),
            sentiment_score=collected_data.get('sentiment', {}).get(sector, 0.0)
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
            
    def _train_final_models(self, X_all: List, y_short_all: List, y_long_all: List):
        """Train final models with processed data"""
        X_all = np.array(X_all)
        y_short_all = np.array(y_short_all)
        y_long_all = np.array(y_long_all)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_all)
        
        # Train models
        print("Training models with enhanced features...")
        self.short_term_model.fit(X_scaled, y_short_all)
        self.long_term_model.fit(X_scaled, y_long_all)
        
        # Store feature importance
        feature_names = ['return_1d', 'return_5d', 'return_20d', 'return_60d',
                        'volume_ratio', 'volume_trend', 'volatility_5d', 'volatility_20d',
                        'RSI', 'MACD', 'MACD_Signal', 'BB_width', 'sentiment']
        
        self.feature_importance = dict(zip(feature_names, 
                                         self.short_term_model.feature_importances_))
        
        print("Model training completed!")
        
    def predict(self, ticker: str, sector: str = None) -> Dict[str, Any]:
        """Make predictions for any ticker (ETF or individual stock)"""
        # Determine if ticker is ETF or individual stock
        is_etf = ticker in self.sector_etfs.values()
        if not is_etf and not any(ticker in stocks for stocks in self.sector_tickers.values()):
            raise ValueError(f"Ticker {ticker} not found in supported tickers")
            
        # If sector not provided, find it
        if sector is None:
            if is_etf:
                sector = next(k for k, v in self.sector_etfs.items() if v == ticker)
            else:
                sector = next(k for k, v in self.sector_tickers.items() if ticker in v)
        
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
            stock_data=stock_data,
            economic_data=collected_data.get('economic_indicators'),
            sentiment_score=collected_data.get('sentiment', {}).get(sector, 0.0)
        )
        
        # Get latest feature values
        latest_features = features.iloc[-1:].values
        
        # Scale features
        X_scaled = self.scaler.transform(latest_features)
        
        # Make predictions
        short_term_prob = self.short_term_model.predict_proba(X_scaled)[0]
        long_term_prob = self.long_term_model.predict_proba(X_scaled)[0]
        
        return {
            'ticker': ticker,
            'sector': sector,
            'type': 'ETF' if is_etf else 'Stock',
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
                'BB_width': float(features['BB_width'].iloc[-1]),
                'volatility': float(features['volatility_20d'].iloc[-1])
            },
            'feature_importance': self.feature_importance
        }
        
    def generate_portfolio_recommendation(self, risk_tolerance: str = 'moderate') -> Dict[str, Any]:
        """Generate comprehensive portfolio recommendations including both ETFs and stocks"""
        risk_weights = {
            'conservative': {'short_term': 0.2, 'long_term': 0.8, 'etf_weight': 0.7},
            'moderate': {'short_term': 0.4, 'long_term': 0.6, 'etf_weight': 0.5},
            'aggressive': {'short_term': 0.6, 'long_term': 0.4, 'etf_weight': 0.3}
        }
        
        weights = risk_weights.get(risk_tolerance, risk_weights['moderate'])
        
        recommendations = {
            'etfs': {'top': [], 'neutral': [], 'avoid': []},
            'stocks': {'top': [], 'neutral': [], 'avoid': []},
            'sector_allocation': {},
            'risk_profile': risk_tolerance
        }
        
        # Process ETFs
        for sector, etf in self.sector_etfs.items():
            pred = self.predict(etf, sector)
            if pred:
                self._categorize_prediction(pred, recommendations['etfs'], weights)
                
        # Process individual stocks
        for sector, tickers in self.sector_tickers.items():
            sector_score = 0
            for ticker in tickers:
                pred = self.predict(ticker, sector)
                if pred:
                    self._categorize_prediction(pred, recommendations['stocks'], weights)
                    sector_score += self._calculate_score(pred, weights)
            
            # Calculate sector allocation
            recommendations['sector_allocation'][sector] = {
                'score': sector_score / len(tickers),
                'suggested_weight': max(0, min(0.3, (sector_score / len(tickers) + 1) / 2))
            }
                
        return recommendations

    def _categorize_prediction(self, pred: Dict[str, Any], target: Dict[str, List], weights: Dict[str, float]):
        """Categorize a prediction into top/neutral/avoid"""
        score = self._calculate_score(pred, weights)
        
        result = {
            'ticker': pred['ticker'],
            'sector': pred['sector'],
            'score': score,
            'short_term': pred['short_term'],
            'long_term': pred['long_term'],
            'technical_indicators': pred['technical_indicators']
        }
        
        if score > 0.3:
            target['top'].append(result)
        elif score < -0.3:
            target['avoid'].append(result)
        else:
            target['neutral'].append(result)
            
    def _calculate_score(self, pred: Dict[str, Any], weights: Dict[str, float]) -> float:
        """Calculate overall score for a prediction"""
        short_score = (1 if pred['short_term']['prediction'] == 'Bullish' else 
                      -1 if pred['short_term']['prediction'] == 'Bearish' else 0)
        long_score = (1 if pred['long_term']['prediction'] == 'Bullish' else 
                     -1 if pred['long_term']['prediction'] == 'Bearish' else 0)
        
        # Technical score
        tech_score = 0
        if 30 <= pred['technical_indicators']['RSI'] <= 70:  # Not overbought/oversold
            tech_score += 0.5
        if pred['technical_indicators']['MACD'] > 0:  # Positive MACD
            tech_score += 0.5
            
        return (short_score * weights['short_term'] * pred['short_term']['confidence'] +
                long_score * weights['long_term'] * pred['long_term']['confidence'] +
                tech_score * 0.2)  # Technical analysis weight

def save_models(predictor: AdvancedStockPredictor, base_path: str = '.'):
    """Save trained models and scaler"""
    joblib.dump(predictor.short_term_model, f'{base_path}/stock_short_term_model.pkl')
    joblib.dump(predictor.long_term_model, f'{base_path}/stock_long_term_model.pkl')
    joblib.dump(predictor.scaler, f'{base_path}/stock_scaler.pkl')

def load_models(predictor: AdvancedStockPredictor, base_path: str = '.'):
    """Load trained models and scaler"""
    predictor.short_term_model = joblib.load(f'{base_path}/stock_short_term_model.pkl')
    predictor.long_term_model = joblib.load(f'{base_path}/stock_long_term_model.pkl')
    predictor.scaler = joblib.load(f'{base_path}/stock_scaler.pkl') 

if __name__ == "__main__":
    # Initialize predictor
    print("Initializing Advanced Stock Predictor...")
    predictor = AdvancedStockPredictor()
    
    # Load pre-trained models
    print("Loading pre-trained models...")
    load_models(predictor)
    
    # Generate predictions for all sectors
    print("\nGenerating predictions for all sectors...")
    
    for sector, stocks in predictor.sector_tickers.items():
        print(f"\n=== {sector} Sector ===")
        
        # First show sector ETF
        etf = predictor.sector_etfs[sector]
        try:
            pred = predictor.predict(etf, sector)
            print(f"\n{etf} (Sector ETF):")
            print(f"Short-term (3m): {pred['short_term']['prediction']} (Confidence: {pred['short_term']['confidence']:.2f})")
            print(f"Long-term (1y): {pred['long_term']['prediction']} (Confidence: {pred['long_term']['confidence']:.2f})")
            print(f"Technical - RSI: {pred['technical_indicators']['RSI']:.2f}, MACD: {pred['technical_indicators']['MACD']:.2f}")
        except Exception as e:
            print(f"Error predicting {etf}: {str(e)}")
        
        # Then show individual stocks
        for ticker in stocks:
            try:
                pred = predictor.predict(ticker, sector)
                print(f"\n{ticker}:")
                print(f"Short-term (3m): {pred['short_term']['prediction']} (Confidence: {pred['short_term']['confidence']:.2f})")
                print(f"Long-term (1y): {pred['long_term']['prediction']} (Confidence: {pred['long_term']['confidence']:.2f})")
                print(f"Technical - RSI: {pred['technical_indicators']['RSI']:.2f}, MACD: {pred['technical_indicators']['MACD']:.2f}")
            except Exception as e:
                print(f"Error predicting {ticker}: {str(e)}")
    
    # Generate portfolio recommendation
    print("\n=== Portfolio Recommendations ===")
    portfolio = predictor.generate_portfolio_recommendation(risk_tolerance='moderate')
    
    # Print sector allocations
    print("\nRecommended Sector Allocations:")
    for sector, allocation in portfolio['sector_allocation'].items():
        print(f"{sector}: {allocation['suggested_weight']*100:.1f}%")
    
    # Print top picks by sector
    print("\nTop Picks by Sector:")
    sector_picks = {}
    for stock in portfolio['stocks']['top']:
        if stock['sector'] not in sector_picks:
            sector_picks[stock['sector']] = []
        sector_picks[stock['sector']].append((stock['ticker'], stock['score']))
    
    for sector, picks in sector_picks.items():
        print(f"\n{sector}:")
        for ticker, score in sorted(picks, key=lambda x: x[1], reverse=True)[:2]:  # Show top 2 per sector
            print(f"{ticker}: Score = {score:.2f}") 