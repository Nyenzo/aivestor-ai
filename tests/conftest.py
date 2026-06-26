import os
import sys
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure AI test mode with lightweight stubs
os.environ.setdefault('AI_TEST_MODE', '1')
os.environ.setdefault('JWT_SECRET', 'test-secret-change-before-production-32')
os.environ.setdefault('FRED_API_KEY', 'mock-fred-key')
os.environ.setdefault('NEWSAPI_KEY', 'mock-news-key')
os.environ.setdefault('DEEPSEEK_API_KEY', 'mock-deepseek-key')

import pandas as pd
from unittest.mock import MagicMock, patch
import yfinance as yf
import requests

# Patch everything BEFORE app is imported
mock_history = pd.DataFrame({
    'Open': [150.0] * 20,
    'High': [155.0] * 20,
    'Low': [145.0] * 20,
    'Close': [152.0] * 20,
    'Volume': [1000000] * 20
}, index=pd.date_range(start=pd.Timestamp.now() - pd.Timedelta(days=365), periods=20, name='Date'))

class MockTicker:
    def __init__(self, *args, **kwargs): 
        self.info = {'symbol': 'MOCK'}
    def history(self, *args, **kwargs): return mock_history

yf.Ticker = MockTicker

class MockResponse:
    def __init__(self):
        self.status_code = 200
    def json(self):
        return {
            'observations': [{'value': '2.5'}],
            'articles': [{'title': 'Bullish market', 'description': 'Growth expected'}]
        }

requests.get = lambda *args, **kwargs: MockResponse()

try:
    from google import genai
except ImportError:
    class _GenAI:
        pass
    genai = _GenAI()

class MockModels:
    def generate_content(self, *args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.text = "This is a mocked Gemini response for testing."
        return mock_resp

class MockClient:
    def __init__(self, *args, **kwargs): 
        self.models = MockModels()

genai.Client = MockClient

mock_sp500_df = pd.DataFrame({'Symbol': ['AAPL', 'MSFT', 'GOOGL']})
pd.read_html = lambda *args, **kwargs: [mock_sp500_df]

try:
    import nltk
    nltk.download = lambda *args, **kwargs: True
    # Pre-emptively mock data.find to avoid throwing exception in chatbot.py
    nltk.data.find = lambda *args, **kwargs: True
except ImportError:
    pass

try:
    from firebase_admin import firestore
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_doc = MagicMock()
    mock_doc.to_dict.return_value = {"question": "Q", "answer": "Mocked FAQ Answer"}
    mock_collection.stream.return_value = [mock_doc]
    mock_db.collection.return_value = mock_collection
    firestore.client = lambda *args, **kwargs: mock_db
except ImportError:
    pass

# Mock AdvancedStockPredictor to bypass pickle load errors
from stock_predictor import AdvancedStockPredictor
# Helper to inject intentional 500 errors for coverage
def _mock_predict(self, ticker):
    if ticker == 'ERROR_TICKER':
        raise Exception("Intentional mock error")
    return {
        'ticker': ticker,
        'current_price': 150.0,
        'price_change_percent': 1.5,
        'short_term_prediction': 'Buy',
        'short_term_probabilities': {'Sell': 0.1, 'Buy': 0.8, 'Hold': 0.1},
        'long_term_prediction': 'Hold',
        'long_term_probabilities': {'Sell': 0.3, 'Buy': 0.3, 'Hold': 0.4},
        'explanation': 'Mocked'
    }

AdvancedStockPredictor.predict = _mock_predict

def _mock_predict_sector(self, sector):
    if sector == 'ERROR_SECTOR':
        raise Exception("Intentional mock error")
    return {
        'sector': sector,
        'short_term_prediction': 'Buy',
        'short_term_probabilities': {'Sell': 0.1, 'Buy': 0.8, 'Hold': 0.1},
        'long_term_prediction': 'Hold',
        'long_term_probabilities': {'Sell': 0.3, 'Buy': 0.3, 'Hold': 0.4},
        'explanation': 'Mocked'
    }

AdvancedStockPredictor.predict_sector = _mock_predict_sector

def _mock_predict_and_output(self, tickers):
    if 'ERROR_TICKER' in tickers:
        raise Exception("Intentional mock error")
    return [_mock_predict(self, t) for t in tickers]

AdvancedStockPredictor.predict_and_output = _mock_predict_and_output
def _mock_generate_portfolio(self, tickers, risk):
    if 'ERROR_TICKER' in tickers:
        raise Exception("Intentional mock error")
    return {
        'portfolio': tickers,
        'allocations': {t: 1.0/len(tickers) for t in tickers},
        'risk_tolerance': risk,
        'explanation': 'Mocked'
    }

AdvancedStockPredictor.generate_portfolio_recommendation = _mock_generate_portfolio
AdvancedStockPredictor.get_top_gainers_losers = lambda self, top_n=3: {
    'gainers': [{'ticker': t, 'change_percent': 5.0} for t in ['AAPL', 'MSFT', 'GOOGL'][:top_n]],
    'losers': [{'ticker': t, 'change_percent': -5.0} for t in ['TSLA', 'META', 'AMZN'][:top_n]]
}

from app import app as flask_app  # noqa: E402

@pytest.fixture()
def client():
    flask_app.config.update({
        'TESTING': True,
    })
    with flask_app.test_client() as client:
        yield client
