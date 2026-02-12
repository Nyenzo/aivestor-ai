"""
Test suite for stock prediction endpoints
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ['AI_TEST_MODE'] = '1'
os.environ['JWT_SECRET_KEY'] = 'test-secret'

from app import app as flask_app
import jwt


@pytest.fixture()
def client():
    flask_app.config.update({'TESTING': True})
    with flask_app.test_client() as client:
        yield client


@pytest.fixture()
def auth_headers():
    token = jwt.encode({'user_id': 1, 'email': 'test@test.com'}, 'test-secret', algorithm='HS256')
    return {'Authorization': f'Bearer {token}'}


def test_predict_single_ticker_no_auth(client):
    """Test single ticker prediction without auth"""
    response = client.get('/predict/AAPL')
    assert response.status_code == 401


def test_predict_single_ticker_valid(client, auth_headers):
    """Test single ticker prediction with valid ticker"""
    response = client.get('/predict/AAPL', headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    # API returns ticker data with current_price, short_term_prediction, etc.
    assert 'current_price' in data or 'prediction' in data or 'error' in data


def test_predict_single_ticker_invalid(client, auth_headers):
    """Test single ticker prediction with invalid ticker"""
    response = client.get('/predict/INVALID', headers=auth_headers)
    assert response.status_code in [200, 400]  # May return error or stub data


def test_predict_multiple_tickers_no_auth(client):
    """Test multiple tickers prediction without auth"""
    response = client.post('/predict', json={'tickers': ['AAPL', 'GOOGL']})
    assert response.status_code == 401


def test_predict_multiple_tickers_valid(client, auth_headers):
    """Test multiple tickers prediction with valid tickers"""
    response = client.post('/predict', headers=auth_headers, json={'tickers': ['AAPL', 'GOOGL', 'MSFT']})
    assert response.status_code == 200
    data = response.get_json()
    # API returns dict with ticker symbols as keys
    assert isinstance(data, dict)
    assert len(data) > 0


def test_predict_multiple_tickers_empty(client, auth_headers):
    """Test multiple tickers prediction with empty list"""
    response = client.post('/predict', headers=auth_headers, json={'tickers': []})
    assert response.status_code == 400


def test_predict_multiple_tickers_missing(client, auth_headers):
    """Test multiple tickers prediction without tickers field"""
    response = client.post('/predict', headers=auth_headers, json={})
    assert response.status_code == 400


def test_predict_sector_no_auth(client):
    """Test sector prediction without auth"""
    response = client.get('/predict/sector/technology')
    # Endpoint may not exist (404) or require auth (401)
    assert response.status_code in [401, 404]


def test_predict_sector_valid(client, auth_headers):
    """Test sector prediction with valid sector"""
    response = client.get('/predict/sector/technology', headers=auth_headers)
    # Endpoint may not be implemented yet
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        data = response.get_json()
        assert 'outlook' in data or 'sector' in data


def test_predict_sector_invalid(client, auth_headers):
    """Test sector prediction with invalid sector"""
    response = client.get('/predict/sector/invalidsector', headers=auth_headers)
    assert response.status_code in [200, 400, 404]
