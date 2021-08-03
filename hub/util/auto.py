import glob
import os, random
from collections import Counter


def detect_compression(path: str):
    """Detects the most frequent compression type of files in a dataset.

    Args:
        path (str): Path to the directory of a local dataset.

    Returns:
        compression (str): Compression type of a dataset.
    """

    types = (".jpeg", ".png", ".jpg")
    file_names = []

    g = glob.glob(os.path.join(path, "**"), recursive=True)

    for name in g:
        if name.endswith(types):
            file_names.append(name)

    if len(file_names) < 100:
        file_names = file_names
    else:
        file_names = random.sample(file_names, 100)

    extension_list = []

    for file in range(len(file_names)):
        extension_list.append(os.path.splitext(file_names[file])[1])

    most_common_extension = [
        extension
        for extension, extension_count in Counter(extension_list).most_common(3)
    ]
    compression = most_common_extension[0].split(".")[1]

    return compression
