import pytest
from api.index import app
import time

@pytest.fixture
def client():
    app.config['TESTING'] = True
    from api.index import ip_tracker, early_block_tracker
    ip_tracker.clear()
    early_block_tracker.clear()
    with app.test_client() as client:
        yield client

def test_method_log_bombing(client, caplog):
    caplog.clear()
    response = client.open('/', method='X'*1000, environ_base={'REMOTE_ADDR': '192.168.0.3'})
    assert response.status_code == 405
    assert len(caplog.records) == 1
    assert len(caplog.records[0].message) < 500, f"Log message is too long ({len(caplog.records[0].message)} chars), log bombing detected!"
