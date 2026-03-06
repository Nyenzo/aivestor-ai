import os
import jwt

def test_protected_requires_token(client):
    resp = client.get('/predict/AAPL')
    assert resp.status_code == 401

def test_protected_accepts_valid_token(client):
    secret = os.environ['JWT_SECRET']
    token = jwt.encode({'service': 'backend'}, secret, algorithm='HS256')
    resp = client.get('/predict/AAPL', headers={'Authorization': f'Bearer {token}'})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['ticker'] == 'AAPL'

def test_app_404_handler(client):

    response = client.get('/some-invalid-url-that-does-not-exist')
    assert response.status_code == 404
