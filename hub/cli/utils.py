import os
import hub
from outdated import check_outdated
import subprocess
import pkg_resources

from hub.exceptions import HubException
from hub.log import logger


def get_cli_version():
    return "1.0.0"
