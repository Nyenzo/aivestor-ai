# Core class for stock price prediction and portfolio recommendations with expanded sectors
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from enhanced_data_collection import EnhancedDataCollector
from process_enhanced_data import EnhancedDataProcessor

class AdvancedStockPredictor:
    def __init__(self):
        # Initialize sector ETFs and tickers, including Streaming Services and deduplicated Commodities
        self.sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financials': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Industrials': 'XLI',
            'Utilities': 'XLU',
            'Materials': 'XLB',
            'Communication Services': 'XLC',
            'Streaming Services': 'FDN',
            'Real Estate': 'XLRE',
            'Commodities_Gold': 'GLD',
            'Commodities_Oil': 'USO',
            'Commodities_Silver': 'SLV',
            'Retail': 'XRT'
        }
        self.sector_tickers = {
            'Technology': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'ADBE'],
            'Healthcare': ['UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY'],
            'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C'],
            'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX'],
            'Consumer Staples': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'MDLZ'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'OXY', 'MPC'],
            'Industrials': ['CAT', 'BA', 'HON', 'UNP', 'MMM', 'GE'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'EXC', 'AEP'],
            'Materials': ['DOW', 'NUE', 'FCX', 'APD', 'ECL'],
            'Communication Services': ['T', 'VZ', 'TMUS', 'LUMN', 'CHTR'],
            'Streaming Services': ['DIS', 'NFLX', 'SPOT', 'ROKU', 'WBD'],
            'Real Estate': ['SPG', 'AMT', 'PLD', 'CCI', 'EQIX'],
            'Commodities_Gold': ['NEM', 'GOLD', 'WPM', 'RGLD', 'FNV', 'XAUUSD'],
            'Commodities_Oil': ['EOG', 'PXD', 'DVN', 'FANG', 'HES'],  # Deduplicated
            'Commodities_Silver': ['PAAS', 'AG', 'SVM', 'HL', 'EXK'],
            'Retail': ['WMT', 'TGT', 'COST', 'LOW', 'TJX']
        }
        self.short_term_model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
        )
        self.long_term_model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
        )
        self.scaler = StandardScaler()
        self.data_collector = EnhancedDataCollector()
        self.data_processor = EnhancedDataProcessor()
        self.feature_importance = {}

    def get_all_tickers(self) -> List[str]:
        # Retrieve all tickers across sectors and ETFs
        tickers = list(self.sector_etfs.values())
        for stocks in self.sector_tickers.values():
            tickers.extend(stocks)
        return list(set(tickers))

    def train_models(self, start_date: str = '2018-01-01'):
        # Train short-term and long-term models using enhanced data
        print("Collecting enhanced historical and fundamental data...")
        collected_data = self.data_collector.collect_all_data(start_date=start_date)
        print("\nProcessing collected data...")
        processed_features = self.data_processor.prepare_features(collected_data)
        
        X_all = []
        y_short_all = []
        y_long_all = []
        feature_names = []
        
        print("\nPreparing training data...")
        for ticker, features_df in processed_features.items():
            if len(features_df) < 252:
                continue
            if not feature_names:
                feature_names = [col for col in features_df.columns 
                               if col not in ['Date', 'Type', 'Sector', 'Open', 'High', 'Low', 'Close', 'Volume']]
            X = features_df[feature_names].values
            short_term_returns = features_df['Close'].pct_change(periods=63)
            long_term_returns = features_df['Close'].pct_change(periods=252)
            valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(short_term_returns) | np.isnan(long_term_returns))
            if valid_idx.sum() > 0:
                X_all.extend(X[valid_idx])
                y_short_all.extend((short_term_returns[valid_idx] > 0).astype(int))
                y_long_all.extend((long_term_returns[valid_idx] > 0).astype(int))
        
        X_all = np.array(X_all)
        y_short_all = np.array(y_short_all)
        y_long_all = np.array(y_long_all)
        
        print("\nScaling features and training models...")
        X_scaled = self.scaler.fit_transform(X_all)
        
        self.short_term_model.fit(X_scaled, y_short_all)
        self.long_term_model.fit(X_scaled, y_long_all)
        
        self.feature_importance = dict(zip(feature_names, 
                                         self.short_term_model.feature_importances_))
        
        print("\nModel training completed!")
        print("\nTop 10 most important features:")
        sorted_features = sorted(self.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)[:10]
        for feature, importance in sorted_features:
            print(f"{feature}: {importance:.4f}")

    def predict(self, ticker: str, sector: str = None) -> Dict[str, Any]:
        # Generate predictions for any ticker with fallback for unknown tickers
        collected_data = self.data_collector.collect_all_data(
            tickers=[ticker],
            start_date=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        )
        processed_features = self.data_processor.prepare_features(collected_data)
        
        if ticker not in processed_features:
            raise ValueError(f"Unable to fetch data for ticker {ticker}")
        
        features_df = processed_features[ticker]
        feature_names = [col for col in features_df.columns 
                        if col not in ['Date', 'Type', 'Sector', 'Open', 'High', 'Low', 'Close', 'Volume']]
        X = features_df[feature_names].iloc[-1:].values
        X_scaled = self.scaler.transform(X)
        
        short_term_prob = self.short_term_model.predict_proba(X_scaled)[0]
        long_term_prob = self.long_term_model.predict_proba(X_scaled)[0]
        signals = self.data_processor.generate_signals({ticker: features_df})
        
        return {
            'ticker': ticker,
            'sector': sector or 'Unknown',
            'short_term': {
                'prediction': 'bullish' if short_term_prob[1] > 0.5 else 'bearish',
                'confidence': float(max(short_term_prob))
            },
            'long_term': {
                'prediction': 'bullish' if long_term_prob[1] > 0.5 else 'bearish',
                'confidence': float(max(long_term_prob))
            },
            'technical_indicators': {
                'RSI': float(features_df['RSI'].iloc[-1]) if 'RSI' in features_df else None,
                'MACD': float(features_df['MACD'].iloc[-1]) if 'MACD' in features_df else None,
                'VIX': float(features_df['VIX'].iloc[-1]) if 'VIX' in features_df else None
            },
            'signals': signals[ticker],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def generate_portfolio_recommendation(self, risk_tolerance: str = 'moderate') -> Dict[str, Any]:
        # Generate sector-based portfolio recommendations based on risk tolerance
        sector_scores = {}
        for sector, etf in self.sector_etfs.items():
            prediction = self.predict(etf, sector)
            short_term_score = prediction['short_term']['confidence'] if prediction['short_term']['prediction'] == 'bullish' else -prediction['short_term']['confidence']
            long_term_score = prediction['long_term']['confidence'] if prediction['long_term']['prediction'] == 'bullish' else -prediction['long_term']['confidence']
            technical_score = sum([1 if signal == 'buy' else -1 if signal == 'sell' else 0 
                                 for signal in prediction['signals'].values()])
            sector_scores[sector] = 0.5 * short_term_score + 0.3 * long_term_score + 0.2 * technical_score
        
        sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
        top_sectors = [s[0] for s in sorted_sectors[:3]]
        neutral_sectors = [s[0] for s in sorted_sectors[3:6]]
        avoid_sectors = [s[0] for s in sorted_sectors[6:]]
        
        weights = {'conservative': 0.2, 'moderate': 0.5, 'aggressive': 0.8}
        weight = weights.get(risk_tolerance, 0.5)
        
        return {
            'risk_tolerance': risk_tolerance,
            'portfolio_weight': weight,
            'top_sectors': top_sectors,
            'neutral_sectors': neutral_sectors,
            'avoid_sectors': avoid_sectors
        }

    def save_models(self, directory: str = 'models'):
        # Save trained models and scaler to disk
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.short_term_model, os.path.join(directory, 'short_term_model.pkl'))
        joblib.dump(self.long_term_model, os.path.join(directory, 'long_term_model.pkl'))
        joblib.dump(self.scaler, os.path.join(directory, 'scaler.pkl'))

def load_models(predictor: 'AdvancedStockPredictor', directory: str = 'models'):
    # Load trained models and scaler from disk
    predictor.short_term_model = joblib.load(os.path.join(directory, 'short_term_model.pkl'))
    predictor.long_term_model = joblib.load(os.path.join(directory, 'long_term_model.pkl'))
    predictor.scaler = joblib.load(os.path.join(directory, 'scaler.pkl'))