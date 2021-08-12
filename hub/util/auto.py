import glob
import os, random
from collections import Counter
from typing import Tuple


def get_most_common_extension(
    path: str, allowed_extensions: Tuple = (".jpeg", ".png", ".jpg")
):
    """Determines the most frequently used extension in a directory of files.

    Args:
        path (str): Directory to scan.
        allowed_extensions (Tuple): File extensions considered for scanning.

    Returns:
        compression (str): Most common extension under the provided path.
    """

    # Return file extension if path is not a directory
    if not os.path.isdir(path):
        file_extension = os.path.splitext(path)[1].split(".")[1]
        if file_extension is None:
            return None
        else:
            return file_extension

    file_names = []

    g = glob.glob(os.path.join(path, "**"), recursive=True)

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


def ingestion_summary(src: str, skipped_files: list):
    """Generate post ingesiton summary in a tree structure.

    Args:
        src (str): Root directory of dataset.
        skipped_files (list): List of files skipped during ingestion.
    """
    columns, lines = os.get_terminal_size()

    mid = int(columns / 2)
    for i in range(columns - 20):
        print("=", end="")
        if i == mid - 10:
            print(" Ingestion Summary ", end="")
    print("\n")

    if not skipped_files:
        print("Ingesiton Complete. No files were skipped.")
        print("\n\n")
        return

    for root, dirs, files in os.walk(src):
        dirs.sort()
        level = root.replace(src, "").count(os.sep)
        indent = " " * 6 * (level)
        print("{}{}/".format(indent, os.path.basename(root)))
        subindent = " " * 6 * (level + 1)
        for f in files:
            if f in skipped_files:
                print("{}[Skipped]  {}".format(subindent, f))
