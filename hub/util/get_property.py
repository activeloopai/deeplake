import re
import os
import pathlib
import hub


def get_property(prop, project):
    """
    Get a property from the `project_name/__init__.py` file.

    Example:
        `get_property("__version__", "hub")`,
        returns the __version__ variable set inside `hub/__init__.py`.
    """

    fname = pathlib.Path(hub.__file__).absolute()
    result = re.search(
        # find variable with name `prop` in the `project + __init__.py` file.
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
        open(fname).read(),
    )
    return result.group(1)
