from hub.util.exceptions import InvalidTagException


def check_tag(tag):
    """Checks whether tag is in the format username/datasetname."""
    if len(tag.split("/")) != 2:
        raise InvalidTagException
