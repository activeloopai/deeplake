from .prettytable import (
    ALL,
    DEFAULT,
    DOUBLE_BORDER,
    FRAME,
    HEADER,
    MARKDOWN,
    MSWORD_FRIENDLY,
    NONE,
    ORGMODE,
    PLAIN_COLUMNS,
    RANDOM,
    SINGLE_BORDER,
    PrettyTable,
    TableHandler,
    from_csv,
    from_db_cursor,
    from_html,
    from_html_one,
    from_json,
)

__all__ = [
    "ALL",
    "DEFAULT",
    "DOUBLE_BORDER",
    "SINGLE_BORDER",
    "FRAME",
    "HEADER",
    "MARKDOWN",
    "MSWORD_FRIENDLY",
    "NONE",
    "ORGMODE",
    "PLAIN_COLUMNS",
    "RANDOM",
    "PrettyTable",
    "TableHandler",
    "from_csv",
    "from_db_cursor",
    "from_html",
    "from_html_one",
    "from_json",
]

try:
    # Python 3.8+
    import importlib.metadata as importlib_metadata
except ImportError:
    # <Python 3.7 and lower
    import importlib_metadata

# __version__ = importlib_metadata.version(__name__)
