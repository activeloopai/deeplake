from typing import List
import hub
from hub.api.dataset import Dataset
from hub.client.client import HubBackendClient


def list_public_datasets(workspace: str = "") -> List:
    """
    Get a list of all public datasets in Platform.

    Args:
        workspace str: Get all public datasets in the given workspace.
            If not given, returns all available public datasets.

    Returns:
        List of dataset names.
    """
    client = HubBackendClient()
    return client.get_public_datasets(workspace=workspace)


def list_my_datasets(workspace: str = "") -> List:
    """
    Get a list of all datasets in the workspace from Platform.

    Args:
        workspace str: Name of the workspace.
            If not given, returns all datasets from the workspace with the same name as the current username.

    Returns:
        List of dataset names.
    """
    client = HubBackendClient()
    return client.get_user_datasets(workspace=workspace)
