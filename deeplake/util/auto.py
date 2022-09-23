import glob
import os, random
from collections import Counter
from typing import Tuple
import shutil
from deeplake.util.exceptions import AutoCompressionError
from deeplake.compression import IMAGE_COMPRESSION_EXTENSIONS


def get_most_common_extension(
    local_path: str, allowed_extensions: Tuple = tuple(IMAGE_COMPRESSION_EXTENSIONS)
):
    """Determines the most frequently used extension in a directory of files.

    Args:
        local_path (str): Directory to scan.
        allowed_extensions (Tuple): File extensions considered for scanning.

    Raises:
        AutoCompressionError: If no files with allowed extensions are found.

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

    g = glob.glob(os.path.join(local_path, "**"), recursive=True)

    file_names = [name for name in g if name.endswith(allowed_extensions)]

    if not file_names:
        raise AutoCompressionError(local_path)

    if len(file_names) > 100:
        file_names = random.sample(file_names, 100)

    extension_list = [
        os.path.splitext(file_names[file])[1] for file in range(len(file_names))
    ]

    most_common_extension = [
        extension
        for extension, extension_count in Counter(extension_list).most_common(3)
    ]
    compression = most_common_extension[0].split(".")[1]

    return compression


def ingestion_summary(local_path: str, skipped_files: list):
    """Generate post ingesiton summary in a tree structure.

    Args:
        local_path (str): Root directory of dataset.
        skipped_files (list): List of files skipped during ingestion.
    """
    columns, lines = shutil.get_terminal_size()

    mid = int(columns / 2)
    print("\n")

    if not skipped_files:
        print("Ingestion Complete. No files were skipped.")
        print("\n")
        return

    at_root = True
    for root, dirs, files in os.walk(local_path):
        files = [f for f in files if not f[0] == "."]
        dirs[:] = [d for d in dirs if not d[0] == "."]
        dirs.sort()

        level = root.replace(local_path, "").count(os.sep)
        indent = " " * 6 * (level)
        if at_root == True:
            print(
                "{}{}/    ".format(
                    indent,
                    os.path.basename(root),
                )
            )
            at_root = False
        else:
            print(
                "{}{}/    ".format(
                    indent,
                    os.path.basename(root),
                )
            )

        subindent = " " * 6 * (level + 1)
        for f in files:
            if f in skipped_files:
                print("{}[Skipped]  {}".format(subindent, f))
