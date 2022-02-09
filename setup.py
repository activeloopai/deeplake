import os
import re
import platform
import subprocess as sp

from setuptools import find_packages, setup

project_name = "hub"


this_directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(this_directory, "hub/requirements/common.txt")) as f:
    requirements = f.readlines()

with open(os.path.join(this_directory, "hub/requirements/tests.txt")) as f:
    tests = f.readlines()

with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


req_map = {
    b: a
    for a, b in (
        re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x.strip("\n"))[0]
        for x in requirements
    )
}

# Add optional dependencies to this dict without version. Version should be specified in requirements.txt
extras = {
    "audio": ["miniaudio"],
    "gcp": ["google-cloud-storage", "google-auth", "google-auth-oauthlib"],
}

all_extras = {r for v in extras.values() for r in v}
non_extra_deps = [req_map[r] for r in req_map if r not in all_extras]
extras["all"] = [req_map[r] for r in all_extras]


init_file = os.path.join(project_name, "__init__.py")


def get_property(prop):
    result = re.search(
        # find variable with name `prop` in the __init__.py file
        rf'{prop}\s*=\s*[\'"]([^\'"]*)[\'"]',
        open(init_file).read(),
    )
    return result.group(1)


def check_ffmpeg():
    ffmpeg_available = True
    print("Checking for FFmpeg")
    try:
        print(sp.check_output(["which", "ffmpeg"]))
    except Exception:
        print("Could not find FFmpeg.")
        ffmpeg_available = False
    return ffmpeg_available


config = {
    "name": project_name,
    "version": get_property("__version__"),
    "description": "Activeloop Hub",
    "long_description": long_description,
    "long_description_content_type": "text/markdown",
    "author": "activeloop.ai",
    "author_email": "support@activeloop.ai",
    "packages": find_packages(),
    "install_requires": requirements,
    "tests_require": tests,
    "include_package_data": True,
    "package_data": {
        "pyffmpeg": [
            "*.c",
            "*.h",
            "libavcodec/*.h",
            "libavformat/*.h",
            "libavutil/*.h",
            "libswscale/*.h",
        ]
    },
    "zip_safe": False,
    "entry_points": {"console_scripts": ["activeloop = hub.cli.commands:cli"]},
    "setup_requires": ["cffi>=1.0.0"],
    "dependency_links": [],
    "project_urls": {
        "Documentation": "https://docs.activeloop.ai/",
        "Source": "https://github.com/activeloopai/Hub",
    },
}


if platform.system() == "Darwin":
    if check_ffmpeg():
        config["cffi_modules"] = ["hub/core/pyffmpeg/build_ffmpeg_ffi.py:ffibuilder"]

setup(**config)
