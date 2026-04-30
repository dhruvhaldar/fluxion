import pytest
from api.index import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_404_error_handler(client):
    response = client.get('/nonexistent_path')
    assert response.status_code == 404
    assert response.headers['Content-Type'] == 'text/plain; charset=utf-8'
    assert b"404 Not Found" in response.data
