import pytest
from api.index import app

def test_payload_log_bombing(caplog):
    app.config['TESTING'] = True
    client = app.test_client()

    caplog.clear()
    for _ in range(50):
        # We need to GET to root, not POST
        response = client.get('/', environ_base={'REMOTE_ADDR': '192.168.0.3'}, data=b"x" * 2000000)
        assert response.status_code == 413
        assert response.headers.get("Connection") == "close"

    assert len(caplog.records) <= 1, f"Log bombing detected! Logs: {len(caplog.records)}"
