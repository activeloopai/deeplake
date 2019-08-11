import logging
import sys

logger = logging.getLogger('hub')
def configure_logger(debug):
    log_level = logging.DEBUG if debug == 1 else logging.INFO
    logging.basicConfig(format='%(message)s',
                        level=log_level,
                        stream=sys.stdout)

