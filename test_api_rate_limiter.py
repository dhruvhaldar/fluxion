import pytest
from api.index import app
import time

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_rate_limiter_allows_requests(client):
    for _ in range(50):
        response = client.get('/', environ_base={'REMOTE_ADDR': '192.168.0.1'})
        assert response.status_code == 200

def test_rate_limiter_blocks_requests(client):
    # Make 100 requests (the limit)
    for _ in range(100):
        response = client.get('/', environ_base={'REMOTE_ADDR': '192.168.0.2'})
        assert response.status_code == 200

    # The 101st request should be blocked
    response = client.get('/', environ_base={'REMOTE_ADDR': '192.168.0.2'})
    assert response.status_code == 429
    assert response.headers['Content-Type'] == 'text/plain; charset=utf-8'
