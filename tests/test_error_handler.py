import pytest
from api.index import app
from werkzeug.exceptions import RequestEntityTooLarge

def test_413_connection_close():
    app.config['TESTING'] = True
    client = app.test_client()

    # Define a temporary route that raises 413
    @app.route('/test_werkzeug_413')
    def trigger_413():
        raise RequestEntityTooLarge()

    response = client.get('/test_werkzeug_413')

    assert response.status_code == 413
    assert response.headers.get("Connection") == "close"
