import os
import pathlib

THIS_FILE = pathlib.Path(__file__).parent.absolute()


def get_dummy_data_path(subpath: str = ""):
    return os.path.join(THIS_FILE, "dummy_data/", subpath)
