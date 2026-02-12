"""
Test suite for chatbot functionality
"""
import os
import sys
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set test mode before imports
os.environ['AI_TEST_MODE'] = '1'
os.environ['JWT_SECRET_KEY'] = 'test-secret'

from app import app as flask_app


@pytest.fixture()
def client():
    flask_app.config.update({'TESTING': True})
    with flask_app.test_client() as client:
        yield client


@pytest.fixture()
def auth_headers():
    """Generate valid JWT token for testing"""
    import jwt
    token = jwt.encode({'user_id': 1, 'email': 'test@test.com'}, 'test-secret', algorithm='HS256')
    return {'Authorization': f'Bearer {token}'}


def test_chatbot_no_auth(client):
    """Test chatbot endpoint without authentication"""
    response = client.post('/chat', json={'query': 'What is Aivestor?'})
    assert response.status_code == 401
    data = response.get_json()
    assert 'error' in data or 'message' in data


def test_chatbot_missing_query(client, auth_headers):
    """Test chatbot endpoint with missing query"""
    response = client.post('/chat', headers=auth_headers, json={})
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data


def test_chatbot_valid_query(client, auth_headers):
    """Test chatbot endpoint with valid query"""
    response = client.post('/chat', headers=auth_headers, json={'query': 'What is Aivestor?'})
    assert response.status_code == 200
    data = response.get_json()
    assert 'answer' in data or 'response' in data
    answer = data.get('answer') or data.get('response')
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_chatbot_investment_query(client, auth_headers):
    """Test chatbot with investment-related query"""
    response = client.post('/chat', headers=auth_headers, json={'query': 'Should I invest in tech stocks?'})
    assert response.status_code == 200
    data = response.get_json()
    assert 'answer' in data or 'response' in data


def test_chatbot_market_query(client, auth_headers):
    """Test chatbot with market information query"""
    response = client.post('/chat', headers=auth_headers, json={'query': 'What are market trends?'})
    assert response.status_code == 200
    data = response.get_json()
    assert 'answer' in data or 'response' in data


def test_chatbot_empty_query(client, auth_headers):
    """Test chatbot with empty query string"""
    response = client.post('/chat', headers=auth_headers, json={'query': ''})
    assert response.status_code == 400


def test_chatbot_long_query(client, auth_headers):
    """Test chatbot with very long query"""
    long_query = 'What should I do with my investments? ' * 50
    response = client.post('/chat', headers=auth_headers, json={'query': long_query})
    assert response.status_code in [200, 400]  # May be truncated or rejected
