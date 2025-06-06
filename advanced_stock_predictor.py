# Importing required libraries for stock prediction and data handling
import pandas as pd
import numpy as np
import pickle
import os
import yfinance as yf
import requests
from typing import List, Dict
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta

# Setting up logging to track API failures
logging.basicConfig(filename='aivestor.log', level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Defining the AdvancedStockPredictor class for stock predictions
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
    SECTOR_KEYWORDS = {
        'Technology': ['tech', 'software', 'hardware', 'semiconductor', 'AI'],
        'Health': ['healthcare', 'pharma', 'biotech', 'medical'],
        'Energy': ['oil', 'gas', 'energy', 'renewable'],
        'Consumer': ['consumer', 'retail', 'e-commerce'],
        'Financials': ['bank', 'finance', 'insurance', 'investment'],
        'Communication': ['telecom', 'media', 'streaming'],
        'Utilities': ['utilities', 'electric', 'water'],
        'Industrials': ['industrial', 'manufacturing', 'construction'],
        'Materials': ['materials', 'chemicals', 'metals'],
        'Consumer Staples': ['staples', 'food', 'beverage'],
        'Retail': ['retail', 'shopping', 'stores'],
        'Real Estate': ['real estate', 'property', 'housing']
    }

    # Initializing the predictor with API keys and sector mappings
    def __init__(self):
        load_dotenv()
        self.FRED_API_KEY = os.getenv('FRED_API_KEY')
        self.NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
        if not self.FRED_API_KEY or not self.NEWSAPI_KEY:
            raise ValueError("FRED_API_KEY or NEWSAPI_KEY not found in .env file")
        self.short_term_model = None
        self.long_term_model = None
        self.scaler = None
        self.sector_mappings = self._create_sector_mappings()
        self.historical_vix = 20.0  # Historical average
        self.historical_economic = None  # Set after first fetch

    # Mapping tickers to their respective sectors
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

    # Loading trained machine learning models and scaler
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

    # Fetching economic indicators from FRED API with fallback handling
    def _fetch_economic_data(self) -> Dict[str, float]:
        try:
            fred_series = {
                'GDP': 'GDP', 'Real_GDP': 'GDPC1', 'Inflation': 'CPIAUCSL',
                'Core_Inflation': 'CPILFESL', 'Unemployment': 'UNRATE',
                'Initial_Claims': 'ICSA', 'Nonfarm_Payrolls': 'PAYEMS',
                'Fed_Funds_Rate': 'FEDFUNDS', '10Y_Treasury': 'DGS10',
                '2Y_Treasury': 'DGS2', 'Industrial_Production': 'INDPRO',
                'Consumer_Sentiment': 'UMCSENT', 'Retail_Sales': 'RSXFS',
                'Housing_Starts': 'HOUST', 'PCE': 'PCE', 'Capacity_Utilization': 'CAPUTL',
                'Labor_Force_Participation': 'CIVPART', 'Yield_Curve_Spread': 'T10Y2Y',
                'GDP_Growth': 'A191RL1Q225SBEA', 'Employment_Change': 'CE16OV'
            }
            data = {}
            for key, series_id in fred_series.items():
                url = f'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={self.FRED_API_KEY}&file_type=json&limit=10&sort_order=desc'
                response = requests.get(url)
                if response.status_code == 200:
                    values = [float(obs['value']) for obs in response.json()['observations']]
                    value = np.mean(values)  # Average of last 10 observations
                    data[key] = value / 100 if 'Rate' in key or 'Spread' in key or 'Inflation' in key else value
                else:
                    logging.warning(f"FRED API failed for {series_id}: {response.status_code}")
                    data[key] = 0.0
            if not self.historical_economic:
                self.historical_economic = data  # Cache first successful fetch
            elif any(v == 0.0 for v in data.values()):
                data = {k: data.get(k, v) if data.get(k, 0.0) != 0.0 else v for k, v in self.historical_economic.items()}
            return data
        except Exception as e:
            logging.warning(f"Error fetching economic data: {e}")
            return self.historical_economic if self.historical_economic else {k: 0.0 for k in fred_series}

    # Fetching VIX data from yfinance with historical fallback
    def _fetch_vix(self) -> float:
        try:
            vix = yf.Ticker('^VIX').history(period='10d')
            if not vix.empty:
                return float(vix['Close'].mean())  # Average of last 10 days
            logging.warning("VIX data empty")
            return self.historical_vix
        except Exception as e:
            logging.warning(f"Error fetching VIX: {e}")
            return self.historical_vix

    # Fetching sector sentiment using News API based on keyword analysis
    def _fetch_sector_sentiment(self, sector: str) -> float:
        try:
            keywords = ' OR '.join(self.SECTOR_KEYWORDS.get(sector, ['sector']))
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            url = f'https://newsapi.org/v2/everything?q={keywords}&from={from_date}&language=en&sortBy=relevancy&apiKey={self.NEWSAPI_KEY}'
            response = requests.get(url)
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                if not articles:
                    return 0.5
                sentiment_score = 0.0
                count = 0
                for article in articles[:10]:  # Limit to 10 articles
                    title = article.get('title', '').lower()
                    desc = article.get('description', '').lower()
                    text = title + ' ' + desc
                    positive_words = ['bullish', 'growth', 'strong', 'rise', 'profit']
                    negative_words = ['bearish', 'decline', 'weak', 'fall', 'loss']
                    score = sum(1 for w in positive_words if w in text) - sum(1 for w in negative_words if w in text)
                    sentiment_score += max(min(score / 5, 1.0), -1.0)  # Normalize to [-1, 1]
                    count += 1
                return (sentiment_score / count + 1) / 2 if count > 0 else 0.5  # Scale to [0, 1]
            logging.warning(f"News API failed for {sector}: {response.status_code}")
            return 0.5
        except Exception as e:
            logging.warning(f"Error fetching sector sentiment for {sector}: {e}")
            return 0.5

    # Computing technical indicators for stock price data
    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            gains = df['Close'].diff().where(lambda x: x > 0, 0).rolling(window=14).mean()
            losses = df['Close'].diff().where(lambda x: x < 0, 0).rolling(window=14).mean()
            df['RSI'] = np.where(losses != 0, 100 - (100 / (1 + gains / losses)), 100 - (100 / (1 + gains / 1e-10)))
            df['RSI'] = df['RSI'].clip(0, 100)
            df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
            df['BB_upper'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
            df['BB_lower'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()
            df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
            df.fillna(0, inplace=True)
            df.replace([np.inf, -np.inf], 0, inplace=True)
            return df
        except Exception as e:
            logging.warning(f"Error computing indicators: {e}")
            return df

    # Retrieving the list of all available tickers
    def get_all_tickers(self) -> List[str]:
        return sorted(list(set(self.TICKERS + list(self.SECTOR_ETFS.values()))))

    # Mapping a ticker to its corresponding sector
    def get_sector(self, ticker: str) -> str:
        return self.sector_mappings.get(ticker, 'Unknown')

    # Predicting sector trends based on ETF and constituent ticker data
    def predict_sector(self, sector: str) -> Dict:
        try:
            if sector not in self.SECTOR_ETFS:
                return {'sector': sector, 'error': 'Invalid sector'}

            etf_ticker = self.SECTOR_ETFS[sector]
            sector_tickers = [t for t, s in self.sector_mappings.items() if s == sector and t in self.TICKERS]
            sector_tickers.append(etf_ticker)

            short_probs = {'Sell': 0.0, 'Buy': 0.0, 'Hold': 0.0}
            long_probs = {'Sell': 0.0, 'Buy': 0.0, 'Hold': 0.0}
            count = 0

            for ticker in sector_tickers:
                pred = self.predict(ticker)
                if 'error' not in pred:
                    for key in short_probs:
                        short_probs[key] += pred['short_term_probabilities'][key]
                        long_probs[key] += pred['long_term_probabilities'][key]
                    count += 1

            if count == 0:
                return {'sector': sector, 'error': 'No valid predictions for sector tickers'}

            for key in short_probs:
                short_probs[key] /= count
                long_probs[key] /= count

            short_pred = max(short_probs, key=short_probs.get)
            long_pred = max(long_probs, key=long_probs.get)

            return {
                'sector': sector,
                'short_term_prediction': short_pred,
                'short_term_probabilities': short_probs,
                'long_term_prediction': long_pred,
                'long_term_probabilities': long_probs,
                'explanation': f"Based on averaged predictions for {sector} tickers, the sector is predicted to {short_pred} in the short term (63 days) and {long_pred} in the long term (252 days)."
            }
        except Exception as e:
            return {'sector': sector, 'error': f'Prediction failed: {e}'}

    # Generating portfolio recommendations based on user risk tolerance
    def generate_portfolio_recommendation(self, tickers: List[str], risk_tolerance: str) -> Dict:
        try:
            valid_tickers = [t for t in tickers if t in self.get_all_tickers()]
            if not valid_tickers:
                return {'portfolio': tickers, 'error': 'No valid tickers provided'}

            risk_weights = {
                'low': {'Buy': 0.6, 'Hold': 0.3, 'Sell': 0.1},
                'medium': {'Buy': 0.4, 'Hold': 0.4, 'Sell': 0.2},
                'high': {'Buy': 0.7, 'Hold': 0.2, 'Sell': 0.1}
            }
            if risk_tolerance.lower() not in risk_weights:
                return {'portfolio': tickers, 'error': 'Invalid risk tolerance'}

            weights = risk_weights[risk_tolerance.lower()]
            allocations = {t: 0.0 for t in valid_tickers}
            short_scores = []
            long_scores = []

            for ticker in valid_tickers:
                pred = self.predict(ticker)
                if 'error' not in pred:
                    short_score = (pred['short_term_probabilities']['Buy'] * weights['Buy'] +
                                   pred['short_term_probabilities']['Hold'] * weights['Hold'] +
                                   pred['short_term_probabilities']['Sell'] * weights['Sell'])
                    long_score = (pred['long_term_probabilities']['Buy'] * weights['Buy'] +
                                  pred['long_term_probabilities']['Hold'] * weights['Hold'] +
                                  pred['long_term_probabilities']['Sell'] * weights['Sell'])
                    short_scores.append(short_score)
                    long_scores.append(long_score)
                else:
                    short_scores.append(0.0)
                    long_scores.append(0.0)

            total_score = sum((s + l) for s, l in zip(short_scores, long_scores) if s > 0 or l > 0)
            if total_score == 0:
                return {'portfolio': tickers, 'error': 'No valid predictions for portfolio'}

            for i, ticker in enumerate(valid_tickers):
                if short_scores[i] > 0 or long_scores[i] > 0:
                    allocations[ticker] = (short_scores[i] + long_scores[i]) / total_score

            return {
                'portfolio': valid_tickers,
                'allocations': allocations,
                'risk_tolerance': risk_tolerance,
                'explanation': f"Portfolio allocations for {len(valid_tickers)} tickers based on {risk_tolerance} risk tolerance, favoring Buy signals for higher risk."
            }
        except Exception as e:
            return {'portfolio': tickers, 'error': f'Recommendation failed: {e}'}

    # Making predictions for a single ticker using real-time data
    def predict(self, ticker: str) -> Dict:
        if not self.short_term_model or not self.long_term_model or not self.scaler:
            self.load_models()

        try:
            df = yf.Ticker(ticker).history(period='1y')
            if df.empty:
                return {'ticker': ticker, 'error': 'No data available'}

            df = self._compute_indicators(df)
            sector = self.get_sector(ticker)
            economic_data = self._fetch_economic_data()
            latest_data = {
                'Close': df['Close'].iloc[-1],
                'RSI': df['RSI'].iloc[-1],
                'MACD': df['MACD'].iloc[-1],
                'BB_upper': df['BB_upper'].iloc[-1],
                'BB_lower': df['BB_lower'].iloc[-1],
                'ATR': df['ATR'].iloc[-1],
                'VIX': self._fetch_vix(),
                'Sector_Sentiment': self._fetch_sector_sentiment(sector),
                **economic_data
            }

            features = pd.DataFrame([latest_data])
            features_scaled = self.scaler.transform(features.values)
            
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
                'explanation': f"Based on technical indicators, VIX, sector sentiment, and economic data, {ticker} is predicted to {signal_map[short_pred]} in the short term (63 days) and {signal_map[long_pred]} in the long term (252 days)."
            }
        except Exception as e:
            return {'ticker': ticker, 'error': f'Prediction failed: {e}'}

    # Predicting and outputting results for multiple tickers
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

# Running predictions for sample tickers if script is executed directly
if __name__ == "__main__":
    predictor = AdvancedStockPredictor()
    predictor.predict_and_output(['AAPL', 'LIN'])