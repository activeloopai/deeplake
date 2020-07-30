import os
import hub
from outdated import check_outdated
import subprocess
import pkg_resources

from hub.exceptions import HubException
from hub.log import logger


def get_cli_version():
    return "1.0.0"


def verify_cli_version():
    try:
        version = pkg_resources.get_distribution(hub.__name__).version
        is_outdated, latest_version = check_outdated(hub.__name__, version)
        if is_outdated:
            print(
                "\033[93m"
                + "Hub is out of date. Please upgrade the package by running `pip3 install --upgrade snark`"
                + "\033[0m"
            )
    except Exception as e:
        logger.error(str(e))


def check_program_exists(command):
    try:
        subprocess.call([command])
    except OSError as e:
        if e.errno == os.errno.ENOENT:
            return False
        else:
            # Something else went wrong while trying to run `wget`
            return True
    return True


def get_proxy_command(proxy):
    ssh_proxy = ""
    if proxy and proxy != " " and proxy != "None" and proxy != "":
        if check_program_exists("ncat"):
            ssh_proxy = '-o "ProxyCommand=ncat --proxy-type socks5 --proxy {} %h %p"'.format(
                proxy
            )
        else:
            raise HubException(
                message="This pod is behind the firewall. You need one more thing. Please install nmap by running `sudo apt-get install nmap` on Ubuntu or `brew install nmap` for Mac"
            )
    return ssh_proxy
