import os
import shutil
import sys
import tempfile
import zlib
import base64
from functools import wraps
from io import BytesIO
from itertools import takewhile
import re


# python3
unicode_type = str
bytes_type = bytes
from urllib.parse import quote, unquote
from pathlib import Path

def unquote_bytes(x):
    return unquote(to_unicode(x))

from shlex import quote as _quote






def url_quote(url):
    """
    Encode a unicode URL to a safe byte string
    """
    # quote() works reliably only with (byte)strings in Python2,
    # hence we need to .encode('utf-8') first. To see by yourself,
    # try quote(u'\xff') in python2. Python3 converts the output
    # always to Unicode, hence we need the outer to_bytes() too.
    #
    # We mark colon as a safe character to keep simple ASCII urls
    # nice looking, e.g. "http://google.com"
    return to_bytes(quote(to_bytes(url), safe='/:'))

def url_unquote(url_bytes):
    """
    Decode a byte string encoded with url_quote to a unicode URL
    """
    return unquote_bytes(url_bytes)

def is_stringish(x):
    """
    Returns true if the object is a unicode or a bytes object
    """
    return isinstance(x, bytes_type) or isinstance(x, unicode_type)


def to_fileobj(x):
    """
    Convert any string-line object to a byte-returning fileobj
    """
    return BytesIO(to_bytes(x))


def to_unicode(x):
    """
    Convert any object to a unicode object
    """
    if isinstance(x, bytes_type):
        return x.decode('utf-8')
    else:
        return unicode_type(x)


def to_bytes(x):
    """
    Convert any object to a byte string
    """
    if isinstance(x, unicode_type):
        return x.encode('utf-8')
    elif isinstance(x, bytes_type):
        return x
    elif isinstance(x, float):
        return repr(x).encode('utf-8')
    else:
        return str(x).encode('utf-8')