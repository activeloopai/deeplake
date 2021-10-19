from importlib.util import find_spec


def pytorch_installed():
    return find_spec("torch") != None


def tensorflow_installed():
    return find_spec("tensorflow") != None


def tfds_installed():
    return find_spec("tensorflow_datasets") != None


def ray_installed():
    return find_spec("ray") != None
