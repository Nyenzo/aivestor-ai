import os
import jwt

def _token():
    return jwt.encode({'service': 'backend'}, os.environ['JWT_SECRET'], algorithm='HS256')

def test_portfolio_recommendation(client):
    resp = client.post('/portfolio', json={
        'tickers': ['SPY','QQQ','VTI'],
        'risk_tolerance': 'medium'
    }, headers={'Authorization': f'Bearer {_token()}'})
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'allocations' in data
    assert set(data['allocations'].keys()) == {'SPY','QQQ','VTI'}

def test_portfolio_no_json(client):

    response = client.post('/portfolio', headers={'Authorization': f'Bearer {_token()}'}, data="not a json")
    assert response.status_code == 500

def test_portfolio_exception(client):

    response = client.post('/portfolio', json={
        'tickers': ['ERROR_TICKER'],
        'risk_tolerance': 'medium'
    }, headers={'Authorization': f'Bearer {_token()}'})
    assert response.status_code == 500
