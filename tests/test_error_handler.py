import pytest
from api.index import handle_exception
from werkzeug.exceptions import RequestEntityTooLarge

def test_413_connection_close():
    e = RequestEntityTooLarge()
    body, code, headers = handle_exception(e)

    assert code == 413
    assert headers.get("Connection") == "close"
