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

def test_rate_limiter_ipv4_mapped_ipv6(client):
    # Make 100 requests using IPv4
    for _ in range(100):
        response = client.get('/', environ_base={'REMOTE_ADDR': '192.168.1.5'})
        assert response.status_code == 200

    # The 101st request using IPv4-mapped IPv6 should be BLOCKED because it's the same logical IP
    response = client.get('/', environ_base={'REMOTE_ADDR': '::ffff:192.168.1.5'})
    assert response.status_code == 429

def test_rate_limiter_lru_eviction(client):
    from api.index import MAX_TRACKED_IPS

    # 1. Add an active IP and consume some quota
    for _ in range(50):
        client.get('/', environ_base={'REMOTE_ADDR': '10.0.0.1'})

    # 2. Fill the IP tracker dictionary just below the capacity limit
    # This loop inserts exactly MAX_TRACKED_IPS - 1 new entries.
    # Total entries = 1 (active IP) + MAX_TRACKED_IPS - 1 = MAX_TRACKED_IPS
    for i in range(2, MAX_TRACKED_IPS + 1):
        client.get('/', environ_base={'REMOTE_ADDR': f'10.0.0.{i}'})

    # 3. Access the active IP again.
    # Because of the LRU eviction policy, 10.0.0.1 will be moved to the end of the dictionary
    client.get('/', environ_base={'REMOTE_ADDR': '10.0.0.1'})

    # 4. Trigger eviction by adding one more new IP
    client.get('/', environ_base={'REMOTE_ADDR': f'10.0.0.{MAX_TRACKED_IPS + 1}'})

    # 5. Because of the LRU policy, 10.0.0.1 should be preserved
    # and not evicted. The oldest of the filler IPs (10.0.0.2) should be evicted instead.
    # Therefore, 10.0.0.1's rate limit quota shouldn't have been reset,
    # but it still has 49 requests left (we used 50 + 1 = 51).
    for _ in range(49):
        response = client.get('/', environ_base={'REMOTE_ADDR': '10.0.0.1'})
        assert response.status_code == 200

    # The 101st request should be blocked
    response = client.get('/', environ_base={'REMOTE_ADDR': '10.0.0.1'})
    assert response.status_code == 429

def test_rate_limiter_ipv6_subnet(client):
    # Make 100 requests using one IP in the /64 subnet
    for _ in range(100):
        response = client.get('/', environ_base={'REMOTE_ADDR': '2001:db8:85a3::1'})
        assert response.status_code == 200

    # The 101st request using a different IP in the SAME /64 subnet should be BLOCKED
    response = client.get('/', environ_base={'REMOTE_ADDR': '2001:db8:85a3::2'})
    assert response.status_code == 429

def test_rate_limiter_ipv6_scope_id(client):
    # Ensure IPv6 addresses with scope IDs are parsed correctly
    # Make 100 requests using an IP with a scope ID
    for _ in range(100):
        response = client.get('/', environ_base={'REMOTE_ADDR': 'fe80::1%eth0'})
        assert response.status_code == 200

    # The 101st request using the same IP should be BLOCKED
    response = client.get('/', environ_base={'REMOTE_ADDR': 'fe80::1%eth0'})
    assert response.status_code == 429

def test_rate_limiter_padded_ip(client):
    # Ensure IPv6 addresses or IPv4 addresses with padding and scope IDs are correctly tracked
    # Make 100 requests using a padded IP with a scope ID
    for _ in range(100):
        response = client.get('/', environ_base={'REMOTE_ADDR': ' 192.168.1.10 %eth0 '})
        assert response.status_code == 200

    # The 101st request using the same padded IP should be BLOCKED
    response = client.get('/', environ_base={'REMOTE_ADDR': ' 192.168.1.10 %eth0 '})
    assert response.status_code == 429
