import pytest
from api.index import app
import time

def test_rate_limiter_log_bombing(caplog):
    app.config['TESTING'] = True
    client = app.test_client()

    # Make 100 requests (the limit)
    for _ in range(100):
        response = client.get('/', environ_base={'REMOTE_ADDR': '192.168.0.3'})
        assert response.status_code == 200

    # The 101st request should be blocked and log a warning
    caplog.clear()
    response = client.get('/', environ_base={'REMOTE_ADDR': '192.168.0.3'})
    assert response.status_code == 429
    assert len(caplog.records) == 1
    assert "Rate limit exceeded" in caplog.records[0].message

    # The 102nd to 150th requests should be blocked but NOT log a warning
    caplog.clear()
    for _ in range(50):
        response = client.get('/', environ_base={'REMOTE_ADDR': '192.168.0.3'})
        assert response.status_code == 429

    assert len(caplog.records) == 0, "Log bombing detected!"
