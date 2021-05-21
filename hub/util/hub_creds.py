import os


def has_hub_testing_creds():
    """Checks if credentials exists"""
    env = os.getenv("ACTIVELOOP_HUB_PASSWORD")
    return env is not None
