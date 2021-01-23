# JUST TO GET COVERAGE
from hub.url import UrlProtocol, UrlType, Url


def test_url():
    Url.parse("Some url")
    Url(
        UrlType.LOCAL,
        UrlProtocol.FILESYSTEM,
        "some path",
        "some bucket",
        "some user",
        "some dataset",
        "some endpoint",
    ).url
