import glob
import os, random
from collections import Counter
from typing import Tuple


def get_most_common_extension(
    local_path: str, allowed_extensions: Tuple = (".jpeg", ".png", ".jpg")
):
    """Determines the most frequently used extension in a directory of files.

    Args:
        local_path (str): Directory to scan.
        allowed_extensions (Tuple): File extensions considered for scanning.

    Returns:
        compression (str): Most common extension under the provided path.
    """

    # Return file extension if path is not a directory
    if not os.path.isdir(local_path):
        file_extension = os.path.splitext(local_path)[1].split(".")[1]
        if file_extension is None:
            return None
        else:
            return file_extension

    file_names = []

    g = glob.glob(os.path.join(local_path, "**"), recursive=True)

    for name in g:
        if name.endswith(allowed_extensions):
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
