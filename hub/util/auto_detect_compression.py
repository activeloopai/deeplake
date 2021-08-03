import glob
import os, random


def auto_detect_compression(path: str):
    """Auto detects the compression type of a dataset using a maximum probable.

    Args:
        path(str): Path to the directory of a local dataset.

    Returns:
        compression(str): Compression type of a dataset
    """

    types = ("*.jpeg", "*.png", "*.jpg")
    list = []
    # if(len(os.listdir(path)))>100:
    #     for file in range(100):
    #         if()
    #         list.append(random.choice(os.listdir(path)))
    # else:
    #     list.append(random.choice(os.listdir(path)))

    os.chdir(path)
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(files, recursive=True))

    print(files_grabbed)


auto_detect_compression(
    "/Users/eshan/Hub/hub/tests/dummy_data/tests_auto/image_classification"
)
