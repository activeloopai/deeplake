import os
import re
import sys

import subprocess
from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext

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
    "audio": ["av"],
    "video": ["av"],
    "av": ["av"],
    "gcp": ["google-cloud-storage", "google-auth", "google-auth-oauthlib"],
    "dicom": ["pydicom"],
    "visualizer": ["IPython", "flask"],
    "gdrive": [
        "google-api-python-client",
        "oauth2client",
        "google-auth",
        "google-auth-oauthlib",
    ],
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


### 
### 
### 
###### Compile pheonix lib
###
###
###

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


# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator:
                try:
                    import ninja  # noqa: F401

                    cmake_args += ["-GNinja"]
                except ImportError:
                    pass

        else:

            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)
        print(["cmake", ext.sourcedir] + cmake_args, build_temp)
        print(["cmake", "--build", "."] + build_args, build_temp)
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)


libpheonix = CMakeExtension('pheonix', sourcedir=os.path.join(this_directory, 'pheonix'))



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
    "cmdclass": {"build_ext": CMakeBuild},
    "project_urls": {
        "Documentation": "https://docs.activeloop.ai/",
        "Source": "https://github.com/activeloopai/Hub",
    },
}

setup(**config)
