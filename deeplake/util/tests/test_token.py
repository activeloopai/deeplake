from hub.util.token import expires_in_to_expires_at, is_expired_token
from time import sleep


def test_expiry():
    creds = {"Authorization": "Bearer 12345", "expires_in": 2}
    expires_in_to_expires_at(creds)
    sleep(2)
    assert is_expired_token(creds)
