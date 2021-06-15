"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

# JUST TO GET COVERAGE
from hub_v1.url import UrlProtocol, UrlType, Url


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
