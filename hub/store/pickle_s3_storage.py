"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from hub.store.s3_storage import S3Storage
import copyreg
from hub.store.s3_storage import S3Storage


def save_s3_storage(obj):
    return obj.__class__, (
        obj.s3fs,
        obj.url,
        obj.public,
        None,
        None,
        None,
        obj.parallel,
        obj.endpoint_url,
    )


copyreg.pickle(S3Storage, save_s3_storage)
