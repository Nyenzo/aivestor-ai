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
import time

# Setting up logging to track API failures
logging.basicConfig(filename='aivestor.log', level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Defining the AdvancedStockPredictor class for stock predictions
class AdvancedStockPredictor:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    CACHE_TTL_SECONDS = int(os.getenv('PREDICTION_CACHE_TTL_SECONDS', '300'))
    MARKET_CACHE_TTL_SECONDS = int(os.getenv('MARKET_CACHE_TTL_SECONDS', '120'))
    ECONOMIC_CACHE_TTL_SECONDS = int(os.getenv('ECONOMIC_CACHE_TTL_SECONDS', '3600'))
    NEWS_CACHE_TTL_SECONDS = int(os.getenv('NEWS_CACHE_TTL_SECONDS', '1800'))
    BASE_FEATURES = ['Close', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'ATR', 'VIX', 'Sector_Sentiment']
    ECONOMIC_FEATURES = [
        'GDP', 'Real_GDP', 'Inflation', 'Core_Inflation', 'Unemployment', 'Initial_Claims',
        'Nonfarm_Payrolls', 'Fed_Funds_Rate', '10Y_Treasury', '2Y_Treasury', 'Industrial_Production',
        'Consumer_Sentiment', 'Retail_Sales', 'Housing_Starts', 'PCE', 'Capacity_Utilization',
        'Labor_Force_Participation', 'Yield_Curve_Spread', 'GDP_Growth', 'Employment_Change'
    ]
    FUNDAMENTAL_FEATURES = ['PE_Ratio', 'EPS', 'Revenue_TTM', 'Debt_to_Equity']
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
        self.short_term_model = None
        self.long_term_model = None
        self.scaler = None
        self.model_mode = 'uninitialized'
        self.model_load_error = None
        self.sector_mappings = self._create_sector_mappings()
        self.historical_vix = 20.0  # Historical average
        self.historical_economic = None  # Set after first fetch
        self._cache = {}
        if not self.FRED_API_KEY:
            logging.warning("FRED_API_KEY not configured; economic features will use neutral cached defaults.")
        if not self.NEWSAPI_KEY:
            logging.warning("NEWSAPI_KEY not configured; sector sentiment will use neutral defaults.")

    def _ttl_get(self, key):
        entry = self._cache.get(key)
        if not entry:
            return None
        expires_at, value = entry
        if expires_at < time.time():
            self._cache.pop(key, None)
            return None
        return value

    def _ttl_set(self, key, value, ttl):
        self._cache[key] = (time.time() + ttl, value)
        return value

    def _feature_names(self, ticker: str) -> List[str]:
        is_etf = ticker in list(self.SECTOR_ETFS.values()) + ['GLD', 'USO', 'XAUUSD']
        names = self.BASE_FEATURES + self.ECONOMIC_FEATURES
        if not is_etf:
            names += self.FUNDAMENTAL_FEATURES
        return names

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
        if self.model_mode != 'uninitialized':
            return
        try:
            with open(os.path.join(self.MODEL_DIR, 'short_term_model.pkl'), 'rb') as f:
                self.short_term_model = pickle.load(f)
            with open(os.path.join(self.MODEL_DIR, 'long_term_model.pkl'), 'rb') as f:
                self.long_term_model = pickle.load(f)
            with open(os.path.join(self.MODEL_DIR, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            self.model_mode = 'persisted-ml'
            logging.info('Loaded persisted prediction models')
        except Exception as error:
            self.short_term_model = None
            self.long_term_model = None
            self.scaler = None
            self.model_mode = 'technical-signal'
            self.model_load_error = type(error).__name__
            logging.warning('Persisted prediction models are unavailable; using technical-signal model (%s)', self.model_load_error)

    def get_model_status(self) -> Dict[str, str]:
        self.load_models()
        return {'mode': self.model_mode, 'artifact_error': self.model_load_error}

    # Fetching economic indicators from FRED API with fallback handling
    def _fetch_economic_data(self) -> Dict[str, float]:
        cached = self._ttl_get('economic_data')
        if cached is not None:
            return cached

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
        if not self.FRED_API_KEY:
            neutral = {key: 0.0 for key in fred_series}
            return self._ttl_set('economic_data', self.historical_economic or neutral, self.ECONOMIC_CACHE_TTL_SECONDS)

        try:
            data = {}
            for key, series_id in fred_series.items():
                url = f'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={self.FRED_API_KEY}&file_type=json&limit=10&sort_order=desc'
                response = requests.get(url, timeout=8)
                if response.status_code == 200:
                    values = [float(obs['value']) for obs in response.json()['observations'] if obs.get('value') not in ('.', None)]
                    value = np.mean(values)  # Average of last 10 observations
                    data[key] = value / 100 if 'Rate' in key or 'Spread' in key or 'Inflation' in key else value
                else:
                    logging.warning(f"FRED API failed for {series_id}: {response.status_code}")
                    data[key] = 0.0
            if not self.historical_economic:
                self.historical_economic = data  # Cache first successful fetch
            elif any(v == 0.0 for v in data.values()):
                data = {k: data.get(k, v) if data.get(k, 0.0) != 0.0 else v for k, v in self.historical_economic.items()}
            return self._ttl_set('economic_data', data, self.ECONOMIC_CACHE_TTL_SECONDS)
        except Exception as e:
            logging.warning(f"Error fetching economic data: {e}")
            return self.historical_economic if self.historical_economic else {k: 0.0 for k in fred_series}

    # Fetching VIX data from yfinance with historical fallback
    def _fetch_vix(self) -> float:
        cached = self._ttl_get('vix')
        if cached is not None:
            return cached
        try:
            vix = yf.Ticker('^VIX').history(period='10d')
            if not vix.empty:
                self.historical_vix = float(vix['Close'].mean())
                return self._ttl_set('vix', self.historical_vix, self.MARKET_CACHE_TTL_SECONDS)
            logging.warning("VIX data empty")
            return self.historical_vix
        except Exception as e:
            logging.warning(f"Error fetching VIX: {e}")
            return self.historical_vix

    # Fetching sector sentiment using News API based on keyword analysis
    def _fetch_sector_sentiment(self, sector: str) -> float:
        cached = self._ttl_get(f'sentiment:{sector}')
        if cached is not None:
            return cached
        if not self.NEWSAPI_KEY:
            return self._ttl_set(f'sentiment:{sector}', 0.5, self.NEWS_CACHE_TTL_SECONDS)
        try:
            keywords = ' OR '.join(self.SECTOR_KEYWORDS.get(sector, ['sector']))
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            url = f'https://newsapi.org/v2/everything?q={keywords}&from={from_date}&language=en&sortBy=relevancy&apiKey={self.NEWSAPI_KEY}'
            response = requests.get(url, timeout=8)
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
                score = (sentiment_score / count + 1) / 2 if count > 0 else 0.5
                return self._ttl_set(f'sentiment:{sector}', score, self.NEWS_CACHE_TTL_SECONDS)
            logging.warning(f"News API failed for {sector}: {response.status_code}")
            return self._ttl_set(f'sentiment:{sector}', 0.5, self.NEWS_CACHE_TTL_SECONDS)
        except Exception as e:
            logging.warning(f"Error fetching sector sentiment for {sector}: {e}")
            return 0.5

    def _fetch_history(self, ticker: str, period: str = '1y') -> pd.DataFrame:
        key = f'history:{ticker}:{period}'
        cached = self._ttl_get(key)
        if cached is not None:
            return cached.copy()
        df = yf.Ticker(ticker).history(period=period)
        return self._ttl_set(key, df.copy(), self.MARKET_CACHE_TTL_SECONDS)

    def _fetch_fundamentals(self, ticker: str) -> Dict[str, float]:
        key = f'fundamentals:{ticker}'
        cached = self._ttl_get(key)
        if cached is not None:
            return cached
        try:
            info = yf.Ticker(ticker).info or {}
            fundamentals = {
                'PE_Ratio': float(info.get('trailingPE') or info.get('forwardPE') or 0.0),
                'EPS': float(info.get('trailingEps') or info.get('forwardEps') or 0.0),
                'Revenue_TTM': float(info.get('totalRevenue') or 0.0),
                'Debt_to_Equity': float(info.get('debtToEquity') or 0.0),
            }
            return self._ttl_set(key, fundamentals, self.ECONOMIC_CACHE_TTL_SECONDS)
        except Exception as e:
            logging.warning(f"Error fetching fundamentals for {ticker}: {e}")
            return {name: 0.0 for name in self.FUNDAMENTAL_FEATURES}

    def _probability_dict(self, model, probabilities) -> Dict[str, float]:
        signal_map = {1: 'Buy', 0: 'Sell', 2: 'Hold'}
        result = {'Sell': 0.0, 'Buy': 0.0, 'Hold': 0.0}
        classes = getattr(model, 'classes_', [0, 1, 2])
        for class_id, probability in zip(classes, probabilities):
            result[signal_map.get(int(class_id), str(class_id))] = float(probability)
        total = sum(result.values()) or 1.0
        return {key: value / total for key, value in result.items()}

    def _align_features(self, features: pd.DataFrame, ticker: str) -> pd.DataFrame:
        ordered = features.reindex(columns=self._feature_names(ticker), fill_value=0.0)
        expected = getattr(self.scaler, 'n_features_in_', ordered.shape[1])
        if ordered.shape[1] < expected:
            for index in range(expected - ordered.shape[1]):
                ordered[f'padding_{index}'] = 0.0
        elif ordered.shape[1] > expected:
            ordered = ordered.iloc[:, :expected]
        return ordered.fillna(0.0).replace([np.inf, -np.inf], 0.0)

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

            risk_tolerance = self._normalize_risk_tolerance(risk_tolerance)

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

    def _normalize_risk_tolerance(self, risk_tolerance) -> str:
        if isinstance(risk_tolerance, (int, float)):
            if risk_tolerance < 0.4:
                return 'low'
            if risk_tolerance > 0.65:
                return 'high'
            return 'medium'
        normalized = str(risk_tolerance or 'medium').strip().lower()
        if normalized in ('conservative', 'low'):
            return 'low'
        if normalized in ('aggressive', 'high'):
            return 'high'
        return 'medium'

    def _technical_signal_prediction(self, ticker: str, df: pd.DataFrame) -> Dict:
        current_price = float(df['Close'].iloc[-1])
        previous_price = float(df['Close'].iloc[-2]) if len(df) > 1 else current_price
        price_change_percent = ((current_price - previous_price) / previous_price) * 100 if previous_price else 0.0
        rsi = float(df['RSI'].iloc[-1])
        macd = float(df['MACD'].iloc[-1])
        prior_macd = float(df['MACD'].iloc[-2]) if len(df) > 1 else macd
        moving_average = float(df['Close'].tail(20).mean())
        score = int(rsi >= 55) - int(rsi <= 45) + int(macd > 0) - int(macd < 0) + int(macd >= prior_macd) - int(macd < prior_macd) + int(current_price >= moving_average) - int(current_price < moving_average)
        action = 'Buy' if score >= 2 else 'Sell' if score <= -2 else 'Hold'
        confidence = min(0.84, max(0.52, 0.54 + abs(score) * 0.07))
        remainder = (1.0 - confidence) / 2
        probabilities = {'Sell': remainder, 'Buy': remainder, 'Hold': remainder}
        probabilities[action] = confidence
        result = {
            'ticker': ticker,
            'current_price': current_price,
            'price_change_percent': float(price_change_percent),
            'short_term_prediction': action,
            'short_term_probabilities': probabilities,
            'long_term_prediction': 'Hold' if action != 'Buy' else 'Buy',
            'long_term_probabilities': probabilities.copy(),
            'confidence': confidence,
            'model_version': 'technical-signal-v1',
            'cache_status': 'miss',
            'explanation': f'Live technical signal uses RSI {rsi:.1f}, MACD direction, and the 20-session trend for {ticker}.',
        }
        return self._ttl_set(f'prediction:{ticker}', result, self.CACHE_TTL_SECONDS)

    # Making predictions for a single ticker using real-time data
    def predict(self, ticker: str) -> Dict:
        ticker = ticker.upper().strip()
        cached = self._ttl_get(f'prediction:{ticker}')
        if cached is not None:
            return {**cached, 'cache_status': 'hit'}

        try:
            self.load_models()
            df = self._fetch_history(ticker, period='1y')
            if df.empty:
                return {'ticker': ticker, 'error': 'No data available'}

            df = self._compute_indicators(df)
            if self.model_mode != 'persisted-ml':
                return self._technical_signal_prediction(ticker, df)
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
            latest_data.update(self._fetch_fundamentals(ticker))

            features = self._align_features(pd.DataFrame([latest_data]), ticker)
            features_scaled = self.scaler.transform(features.values)
            
            short_pred = self.short_term_model.predict(features_scaled)[0]
            short_proba = self.short_term_model.predict_proba(features_scaled)[0]
            long_pred = self.long_term_model.predict(features_scaled)[0]
            long_proba = self.long_term_model.predict_proba(features_scaled)[0]
            
            # Calculate price change percentage
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
            price_change_percent = ((current_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0
            
            signal_map = {1: 'Buy', 0: 'Sell', 2: 'Hold'}
            short_probabilities = self._probability_dict(self.short_term_model, short_proba)
            long_probabilities = self._probability_dict(self.long_term_model, long_proba)
            confidence = max(short_probabilities.values())

            result = {
                'ticker': ticker,
                'current_price': float(current_price),
                'price_change_percent': float(price_change_percent),
                'short_term_prediction': signal_map[short_pred],
                'short_term_probabilities': short_probabilities,
                'long_term_prediction': signal_map[long_pred],
                'long_term_probabilities': long_probabilities,
                'confidence': float(confidence),
                'model_version': 'enhanced-gradient-boosting-v2',
                'cache_status': 'miss',
                'explanation': f"Based on technical indicators, VIX, fundamentals, sector sentiment, and economic data, {ticker} is predicted to {signal_map[short_pred]} in the short term (63 days) and {signal_map[long_pred]} in the long term (252 days)."
            }
            self._ttl_set(f'prediction:{ticker}', result, self.CACHE_TTL_SECONDS)
            return result
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

    def get_top_gainers_losers(self, top_n=3):
        """
        Returns the top N gainers and losers among tracked tickers based on daily percent change.
        """
        import yfinance as yf
        results = []
        for ticker in self.TICKERS:
            try:
                df = self._fetch_history(ticker, period='5d')
                if df.empty or len(df) < 2:
                    continue
                current = df['Close'].iloc[-1]
                prev = df['Close'].iloc[-2]
                change = ((current - prev) / prev) * 100 if prev != 0 else 0
                results.append({
                    'ticker': ticker,
                    'current_price': current,
                    'previous_close': prev,
                    'change_percent': change
                })
            except Exception as e:
                continue
        gainers = sorted(results, key=lambda x: x['change_percent'], reverse=True)[:top_n]
        losers = sorted(results, key=lambda x: x['change_percent'])[:top_n]
        return {'gainers': gainers, 'losers': losers}

# Running predictions for sample tickers if script is executed directly
if __name__ == "__main__":
    predictor = AdvancedStockPredictor()
    predictor.predict_and_output(['AAPL', 'LIN'])
