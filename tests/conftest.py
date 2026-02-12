import os
import sys
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure AI test mode with lightweight stubs
os.environ.setdefault('AI_TEST_MODE', '1')
os.environ.setdefault('JWT_SECRET_KEY', 'test-secret')

from app import app as flask_app  # noqa: E402

@pytest.fixture()
def client():
    flask_app.config.update({
        'TESTING': True,
    })
    with flask_app.test_client() as client:
        yield client
