import pytest
from api.index import app

def test_404_connection_close():
    app.config['TESTING'] = True
    client = app.test_client()
    response = client.get('/nonexistent_for_testing')
    assert response.status_code == 404
    assert response.headers.get('Connection') == 'close', "404 Not Found response should explicitly close the connection"

def test_405_connection_close():
    app.config['TESTING'] = True
    client = app.test_client()
    response = client.post('/')
    assert response.status_code == 405
    assert response.headers.get('Connection') == 'close', "405 Method Not Allowed response should explicitly close the connection"
