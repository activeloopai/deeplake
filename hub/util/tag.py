from hub.util.exceptions import InvalidHubPathException


def check_hub_path(path):
    """Checks whether tag is in the format hub://username/datasetname."""
    tag = path[6:]
    if len(tag.split("/")) != 2:
        raise InvalidHubPathException(path)
