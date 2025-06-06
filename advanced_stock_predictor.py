# Managing stock predictions and outputting results
import pandas as pd
import numpy as np
import pickle
import os
import yfinance as yf
from typing import List, Dict
from sklearn.preprocessing import StandardScaler

class AdvancedStockPredictor:
    MODEL_DIR = 'models'
    TICKERS = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'TSM', 'QCOM',
        'PFE', 'ABBV', 'LLY', 'MRK', 'JNJ', 'T', 'VZ', 'TMUS', 'CMCSA', 'CHTR', 'XOM', 'CVX',
        'COP', 'BP', 'SHEL', 'WMT', 'TGT', 'COST', 'HD', 'LOW', 'JPM', 'BAC', 'WFC', 'C', 'GS',
        'V', 'MA', 'AXP', 'PG', 'KO', 'PEP', 'NKE', 'MCD', 'CAT', 'DE', 'MMM', 'BA', 'GE',
        'NFLX', 'DIS', 'SPOT', 'ROKU', 'LIN', 'SHW', 'FCX', 'ECL', 'GLD', 'USO', 'XAUUSD'
    ]
    SECTOR_ETFS = {
        'Technology': 'XLK',
        'Health': 'XLV',
        'Energy': 'XLE',
        'Consumer': 'XLY',
        'Financials': 'XLF',
        'Communication': 'XLC',
        'Utilities': 'XLU',
        'Industrials': 'XLI',
        'Materials': 'XLB',
        'Consumer Staples': 'XLP',
        'Retail': 'XRT',
        'Real Estate': 'XLRE'
    }

    # Initializing the predictor with ticker and sector mappings
    def __init__(self):
        self.short_term_model = None
        self.long_term_model = None
        self.scaler = None
        self.sector_mappings = self._create_sector_mappings()

    # Creating a mapping of tickers to their respective sectors
    def _create_sector_mappings(self) -> Dict[str, str]:
        mappings = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'AMZN': 'Consumer', 'GOOGL': 'Technology',
            'META': 'Technology', 'TSLA': 'Consumer', 'NVDA': 'Technology', 'AMD': 'Technology',
            'INTC': 'Technology', 'TSM': 'Technology', 'QCOM': 'Technology', 'PFE': 'Health',
            'ABBV': 'Health', 'LLY': 'Health', 'MRK': 'Health', 'JNJ': 'Health', 'T': 'Communication',
            'VZ': 'Communication', 'TMUS': 'Communication', 'CMCSA': 'Communication',
            'CHTR': 'Communication', 'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'BP': 'Energy',
            'SHEL': 'Energy', 'WMT': 'Consumer Staples', 'TGT': 'Consumer Staples',
            'COST': 'Consumer Staples', 'HD': 'Consumer', 'LOW': 'Consumer', 'JPM': 'Financials',
            'BAC': 'Financials', 'WFC': 'Financials', 'C': 'Financials', 'GS': 'Financials',
            'V': 'Financials', 'MA': 'Financials', 'AXP': 'Financials', 'PG': 'Consumer Staples',
            'KO': 'Consumer Staples', 'PEP': 'Consumer Staples', 'NKE': 'Consumer',
            'MCD': 'Consumer', 'CAT': 'Industrials', 'DE': 'Industrials', 'MMM': 'Industrials',
            'BA': 'Industrials', 'GE': 'Industrials', 'NFLX': 'Communication', 'DIS': 'Communication',
            'SPOT': 'Communication', 'ROKU': 'Communication', 'LIN': 'Materials', 'SHW': 'Materials',
            'FCX': 'Materials', 'ECL': 'Materials', 'GLD': 'Materials', 'USO': 'Energy',
            'XAUUSD': 'Materials', 'XLK': 'Technology', 'XLV': 'Health', 'XLE': 'Energy',
            'XLY': 'Consumer', 'XLF': 'Financials', 'XLC': 'Communication', 'XLU': 'Utilities',
            'XLI': 'Industrials', 'XLB': 'Materials', 'XLP': 'Consumer Staples', 'XRT': 'Retail',
            'XLRE': 'Real Estate'
        }
        return mappings

    # Loading trained models and scaler for predictions
    def load_models(self):
        try:
            with open(os.path.join(self.MODEL_DIR, 'short_term_model.pkl'), 'rb') as f:
                self.short_term_model = pickle.load(f)
            with open(os.path.join(self.MODEL_DIR, 'long_term_model.pkl'), 'rb') as f:
                self.long_term_model = pickle.load(f)
            with open(os.path.join(self.MODEL_DIR, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            print("Loaded models and scaler")
        except FileNotFoundError:
            print("Error: Trained models not found. Please run train_enhanced_model_cv.py first.")
            raise

    # Computing technical indicators for a DataFrame
    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df['RSI'] = (df['Close'].diff().where(lambda x: x > 0, 0).rolling(window=14).mean() /
                         df['Close'].diff().where(lambda x: x < 0, 0).rolling(window=14).mean() * -100 + 100)
            df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
            df['BB_upper'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
            df['BB_lower'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()
            df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
            return df
        except Exception as e:
            print(f"Error computing indicators: {e}")
            return df

    # Retrieving all available tickers
    def get_all_tickers(self) -> List[str]:
        return sorted(list(set(self.TICKERS + list(self.SECTOR_ETFS.values()))))

    # Mapping a ticker to its sector
    def get_sector(self, ticker: str) -> str:
        return self.sector_mappings.get(ticker, 'Unknown')

    # Predicting sector trends (placeholder)
    def predict_sector(self, sector: str) -> Dict:
        print(f"Predicting sector {sector}: Not implemented")
        return {'sector': sector, 'prediction': 'Not implemented'}

    # Generating portfolio recommendations (placeholder)
    def generate_portfolio_recommendation(self, tickers: List[str], risk_tolerance: str) -> Dict:
        print(f"Generating portfolio for {tickers} with risk {risk_tolerance}: Not implemented")
        return {'portfolio': tickers, 'recommendation': 'Not implemented'}

    # Making predictions for a ticker
    def predict(self, ticker: str) -> Dict:
        if not self.short_term_model or not self.long_term_model or not self.scaler:
            self.load_models()

        try:
            df = yf.Ticker(ticker).history(period='1y')
            if df.empty:
                return {'ticker': ticker, 'error': 'No data available'}

            df = self._compute_indicators(df)
            latest_data = {
                'Close': df['Close'].iloc[-1],
                'RSI': df['RSI'].iloc[-1],
                'MACD': df['MACD'].iloc[-1],
                'BB_upper': df['BB_upper'].iloc[-1],
                'BB_lower': df['BB_lower'].iloc[-1],
                'ATR': df['ATR'].iloc[-1],
                'VIX': 20.0,  # Placeholder
                'Sector_Sentiment': 0.5,  # Placeholder
                'GDP': 0.02, 'Real_GDP': 0.015, 'Inflation': 0.03, 'Core_Inflation': 0.025,
                'Unemployment': 0.04, 'Initial_Claims': 200000, 'Nonfarm_Payrolls': 150000,
                'Fed_Funds_Rate': 0.05, '10Y_Treasury': 0.042, '2Y_Treasury': 0.04,
                'Industrial_Production': 0.01, 'Consumer_Sentiment': 70.0, 'Retail_Sales': 0.005,
                'Housing_Starts': 1400000, 'PCE': 0.02, 'Capacity_Utilization': 0.75,
                'Labor_Force_Participation': 0.62, 'Yield_Curve_Spread': 0.002,
                'GDP_Growth': 0.02, 'Employment_Change': 0.01
            }

            features = pd.DataFrame([latest_data])
            features_scaled = self.scaler.transform(features.values)  # Convert to NumPy array
            
            short_pred = self.short_term_model.predict(features_scaled)[0]
            short_proba = self.short_term_model.predict_proba(features_scaled)[0]
            long_pred = self.long_term_model.predict(features_scaled)[0]
            long_proba = self.long_term_model.predict_proba(features_scaled)[0]
            
            signal_map = {1: 'Buy', 0: 'Sell', 2: 'Hold'}
            return {
                'ticker': ticker,
                'short_term_prediction': signal_map[short_pred],
                'short_term_probabilities': dict(zip(['Sell', 'Buy', 'Hold'], short_proba)),
                'long_term_prediction': signal_map[long_pred],
                'long_term_probabilities': dict(zip(['Sell', 'Buy', 'Hold'], long_proba)),
                'explanation': f"Based on technical indicators and market data, {ticker} is predicted to {signal_map[short_pred]} in the short term (63 days) and {signal_map[long_pred]} in the long term (252 days)."
            }
        except Exception as e:
            return {'ticker': ticker, 'error': f'Prediction failed: {e}'}

    # Predicting and outputting results for one or multiple tickers
    def predict_and_output(self, tickers: List[str] = None):
        if tickers is None:
            tickers = self.get_all_tickers()
        
        results = []
        for ticker in tickers:
            pred = self.predict(ticker)
            results.append(pred)
            
            print(f"\nPrediction for {ticker}:")
            if 'error' in pred:
                print(f"Error: {pred['error']}")
            else:
                print(f"Short-Term Prediction (63 days): {pred['short_term_prediction']}")
                print(f"Probabilities: Sell={pred['short_term_probabilities']['Sell']:.4f}, Buy={pred['short_term_probabilities']['Buy']:.4f}, Hold={pred['short_term_probabilities']['Hold']:.4f}")
                print(f"Long-Term Prediction (252 days): {pred['long_term_prediction']}")
                print(f"Probabilities: Sell={pred['long_term_probabilities']['Sell']:.4f}, Buy={pred['long_term_probabilities']['Buy']:.4f}, Hold={pred['long_term_probabilities']['Hold']:.4f}")
                print(f"Explanation: {pred['explanation']}")
        
        return results

if __name__ == "__main__":
    # Executing predictions and outputting results for sample tickers
    predictor = AdvancedStockPredictor()
    predictor.predict_and_output(['AAPL', 'LIN'])