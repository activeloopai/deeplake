import os
import re
import sys

import subprocess
from distutils.core import setup, Extension
from setuptools import find_packages # , setup

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
    "video": ["av"],
    "dicom": ["pydicom"],
    "visualizer": ["IPython", "flask"],
}

all_extras = {r for v in extras.values() for r in v}
install_requires = [req_map[r] for r in req_map if r not in all_extras]
extras_require = {k: [req_map[r] for r in v] for k, v in extras.items()}
extras_require["all"] = [req_map[r] for r in all_extras]


init_file = os.path.join(project_name, "__init__.py")


def get_property(prop):
    result = re.search(
        # find variable with name `prop` in the __init__.py file
        rf'{prop}\s*=\s*[\'"]([^\'"]*)[\'"]',
        open(init_file).read(),
    )
    return result.group(1)


# Compile pheonix lib
def pkgconfig(package, kw):
    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    output = subprocess.getoutput(
        'pkg-config --cflags --libs {}'.format(package))
    if 'not found' in output:
        raise Exception(f"Could not find required package: {package}.")
    for token in output.strip().split():
        kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
    return kw


boost_path = "/home/davit/Git/tmp/boost_1_78_0"
aws_sdk_path = "/home/davit/Git/prefetching-experiments/aws_sdk_build/lib"
ldaws_sdk_path = "-L:/home/davit/Git/prefetching-experiments/aws_sdk_build/lib/lib"

if False:
    load(name="pheonix_cpp",
        sources=[os.path.join(module_path, f"{el}") for el in ["libpheonix.cpp"]], #, "scheduler.cpp"]],
        extra_include_paths=[module_path], # boost_path
        extra_cflags=["-fcoroutines", "-std=c++17"],
        extra_ldflags=["-lcurl", 
                    "-laws-cpp-sdk-core", 
                    "-laws-cpp-sdk-s3"
        ],# , "-L:/home/davit/Git/tmp/boost_1_78_0/stage/lib"], 
        build_directory=os.path.join(module_path, "build"),
        verbose=True)
    
from pybind11.setup_helpers import Pybind11Extension, build_ext


sources = ['./pheonix/libpheonix.cpp']
module_path = os.path.join(os.path.dirname(__file__), "pheonix")

static_libraries = ['igraph']
static_lib_dir = '/system/lib'
libraries = ['z', 'xml2', 'gmp']
library_dirs = ['/system/lib', '/system/lib64']
extra_compile_args = ['-std=c++2a', '-fcoroutines']

if sys.platform == 'win32':
    libraries.extend(static_libraries)
    library_dirs.append(static_lib_dir)
    extra_objects = []
else: # POSIX
    extra_objects = ['{}/lib{}.a'.format(static_lib_dir, l) for l in static_libraries]

extension_kwargs = {
    'sources': sources,
    'include_dirs': [module_path],
    # 'libraries': libraries,
    # 'library_dirs': library_dirs,
    # 'extra_objects': extra_objects,
    'extra_compile_args': extra_compile_args,
}
                 
# extension_kwargs = pkgconfig('opencv4', extension_kwargs)
# extension_kwargs = pkgconfig('libturbojpeg', extension_kwargs)

# extension_kwargs['libraries'].append('pthread')


libpheonix = Pybind11Extension('pheonix',
                        **extension_kwargs)



config = {
    "name": project_name,
    "version": get_property("__version__"),
    "description": "Activeloop Hub",
    "long_description": long_description,
    "long_description_content_type": "text/markdown",
    "author": "activeloop.ai",
    "author_email": "support@activeloop.ai",
    "packages": find_packages(),
    "install_requires": install_requires,
    "extras_require": extras_require,
    "tests_require": tests,
    "include_package_data": True,
    "zip_safe": False,
    "entry_points": {"console_scripts": ["activeloop = hub.cli.commands:cli"]},
    "dependency_links": [],
    "ext_modules": [libpheonix],
    "project_urls": {
        "Documentation": "https://docs.activeloop.ai/",
        "Source": "https://github.com/activeloopai/Hub",
    },
}

setup(**config)
