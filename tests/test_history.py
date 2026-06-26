import pytest
import jwt
import os

def _token():
    return jwt.encode({'sub':'abc','uid':'abc'}, os.environ.get('JWT_SECRET', 'test-secret-change-before-production-32'), algorithm='HS256')

@pytest.fixture
def auth_headers():
    return {'Authorization': f'Bearer {_token()}'}

def test_get_history_valid_ticker(client, auth_headers):

    response = client.get('/history/AAPL?period=1y', headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert 'ticker' in data
    assert data['ticker'] == 'AAPL'
    assert 'data' in data
    assert isinstance(data['data'], list)
    if len(data['data']) > 0:
        assert 'date' in data['data'][0]
        assert 'price' in data['data'][0]

def test_get_history_multiple_periods(client, auth_headers):

    periods = ['1mo', '6mo', '5y', 'invalid_period']
    for p in periods:
        response = client.get(f'/history/MSFT?period={p}', headers=auth_headers)
        assert response.status_code == 200
        data = response.get_json()
        assert 'ticker' in data
        assert isinstance(data['data'], list)

def test_get_history_unauthorized(client):

    response = client.get('/history/AAPL')
    assert response.status_code == 401

def test_history_exception(client, auth_headers):

    response = client.get('/history/ERROR_TICKER_HIST?period=1y', headers=auth_headers)
    assert response.status_code == 200
