
import os
import sys
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set test mode before imports
os.environ['AI_TEST_MODE'] = '1'
TEST_JWT_SECRET = 'test-secret-change-before-production-32'
os.environ['JWT_SECRET'] = TEST_JWT_SECRET

from app import app as flask_app

@pytest.fixture()
def client():
    flask_app.config.update({'TESTING': True})
    with flask_app.test_client() as client:
        yield client

@pytest.fixture()
def auth_headers():

    import jwt
    token = jwt.encode({'user_id': 1, 'email': 'test@test.com'}, TEST_JWT_SECRET, algorithm='HS256')
    return {'Authorization': f'Bearer {token}'}

def test_chatbot_no_auth(client):

    response = client.post('/chat', json={'query': 'What is Aivestor?'})
    assert response.status_code == 401
    data = response.get_json()
    assert 'error' in data or 'message' in data

def test_chatbot_missing_query(client, auth_headers):

    response = client.post('/chat', headers=auth_headers, json={})
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data

def test_chatbot_valid_query(client, auth_headers):

    response = client.post('/chat', headers=auth_headers, json={'query': 'What is Aivestor?'})
    assert response.status_code == 200
    data = response.get_json()
    assert 'answer' in data or 'response' in data
    answer = data.get('answer') or data.get('response')
    assert isinstance(answer, str)
    assert len(answer) > 0

def test_chatbot_investment_query(client, auth_headers):

    response = client.post('/chat', headers=auth_headers, json={'query': 'Should I invest in tech stocks?'})
    assert response.status_code == 200
    data = response.get_json()
    assert 'answer' in data or 'response' in data

def test_chatbot_market_query(client, auth_headers):

    response = client.post('/chat', headers=auth_headers, json={'query': 'What are market trends?'})
    assert response.status_code == 200
    data = response.get_json()
    assert 'answer' in data or 'response' in data

def test_chatbot_empty_query(client, auth_headers):

    response = client.post('/chat', headers=auth_headers, json={'query': ''})
    assert response.status_code == 400

def test_chatbot_long_query(client, auth_headers):

    long_query = 'What should I do with my investments? ' * 50
    response = client.post('/chat', headers=auth_headers, json={'query': long_query})
    assert response.status_code in [200, 400]  # May be truncated or rejected

def test_chatbot_tfidf_fallback(client, auth_headers, monkeypatch):

    from app import chatbot
    
    # Temporarily monkeypatch the globally instantiated model to throw an exception
    def mock_broken_generate(*args, **kwargs):
        raise Exception("Gemini API Timeout or Quota Error")
        
    monkeypatch.setattr(chatbot.client.models, 'generate_content', mock_broken_generate)
    
    # Issue a query that forces TF-IDF matching on default FAQs
    response = client.post('/chat', headers=auth_headers, json={'query': 'What is the risk assessment feature?'})
    
    assert response.status_code == 200
    data = response.get_json()
    answer = data.get('answer') or data.get('response', '')
    
    # It should gracefully return an offline answer and not throw 500
    # The default corpus tfidf result or the global fallback "Haha you got me..."
    assert isinstance(answer, str)
    assert len(answer) > 0
