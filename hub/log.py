import logging

logger = logging.getLogger('hub')


def configure_logger(debug=0):
    log_level = logging.DEBUG if debug == 1 else logging.INFO
    logger.setLevel(log_level)


configure_logger(0)
