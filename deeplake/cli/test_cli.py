from deeplake.util.exceptions import LoginException
from deeplake.cli.auth import login, logout
from click.testing import CliRunner

import pytest


def test_bad_token():
    runner = CliRunner()

    result = runner.invoke(login, "-t abcd")
    assert isinstance(result.exception, LoginException)
