import logging

logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("nose").setLevel(logging.WARNING)


def logging_basic_config(**kwargs):
    """Does basic configuration of the logging system."""
    root = logging.getLogger()
    # Clearing handler due to other handler in Hub
    # Later we can update it
    root.handlers.clear()
    if len(root.handlers) == 0:
        # The root logger does not have any handlers yet, so basic configuration is needed
        # Setting custom formatter
        format_string = "[%(asctime)s] [%(levelname)s] %(message)s"
        kwargs.setdefault("format", format_string)
        logging.basicConfig(**kwargs)


def get_package_logger(module_name):
    """Returns the Logger for the coresponding package."""
    package_name = module_name.rpartition(".")[0]
    logger = logging.getLogger(package_name)
    logger.addHandler(logging.NullHandler())
    return logger


logger = get_package_logger(__name__)
