import pytest
from api.index import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    from api.index import ip_tracker, early_block_tracker
    ip_tracker.clear()
    early_block_tracker.clear()
    with app.test_client() as client:
        yield client

def test_http_method_restriction(client, caplog):
    caplog.clear()

    # POST should be blocked
    response = client.post('/', environ_base={'REMOTE_ADDR': '192.168.0.10'})
    assert response.status_code == 405
    assert response.headers['Content-Type'] == 'text/plain; charset=utf-8'
    assert "Blocked request using unsupported method" in caplog.records[0].message

    # OPTIONS should be allowed
    response = client.options('/', environ_base={'REMOTE_ADDR': '192.168.0.10'})
    assert response.status_code == 200 or response.status_code == 404 # 404 if OPTIONS not explicitly handled by route, but shouldn't be 405 from before_request

    # PUT should be blocked
    response = client.put('/', environ_base={'REMOTE_ADDR': '192.168.0.10'})
    assert response.status_code == 405
