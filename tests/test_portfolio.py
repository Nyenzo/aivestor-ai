import os
import jwt

def _token():
    return jwt.encode({'service': 'backend'}, os.environ['JWT_SECRET_KEY'], algorithm='HS256')


def test_portfolio_recommendation(client):
    resp = client.post('/portfolio', json={
        'tickers': ['SPY','QQQ','VTI'],
        'risk_tolerance': 'medium'
    }, headers={'Authorization': f'Bearer {_token()}'})
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'allocation' in data
    assert set(data['allocation'].keys()) == {'SPY','QQQ','VTI'}
