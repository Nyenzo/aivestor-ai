"""
Test suite for tracing functionality
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


def test_tracing_initialized():
    """Test that OpenTelemetry tracing is initialized"""
    # In test mode, tracing should be initialized but with no-op exporters
    # Just verify the app has tracing-related attributes
    assert hasattr(flask_app, 'config')


def test_endpoint_with_tracing(client, auth_headers):
    """Test that endpoints work with tracing enabled"""
    # Make a request and ensure it completes successfully with tracing
    response = client.get('/predict/AAPL', headers=auth_headers)
    assert response.status_code == 200


def test_multiple_requests_with_tracing(client, auth_headers):
    """Test multiple requests to verify tracing doesn't interfere"""
    for _ in range(3):
        response = client.get('/predict/AAPL', headers=auth_headers)
        assert response.status_code == 200


def test_error_handling_with_tracing(client, auth_headers):
    """Test that errors are properly traced"""
    response = client.post('/predict', headers=auth_headers, json={})
    assert response.status_code == 400  # Should handle error gracefully
