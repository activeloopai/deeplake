load("@rules_cc//cc:defs.bzl", "cc_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

"""
cc_binary(
    name = "pheonix",
    srcs = ["eventuals.cpp"],
    deps = [
        "@com_github_3rdparty_eventuals//eventuals","
    ],
    # NOTE: at least on macOS we can't link statically because there
    # are certain system libraries that require dynamic loading.
    linkstatic = False,
)
"""

pybind_extension(
    name = "pheonix",
    srcs = ["libpheonix.cpp", "requests.h", "scheduler.h"],
    deps = [
        "@com_github_3rdparty_eventuals//eventuals",
    ],
    linkopts = ["-lcurl", "-lstdc++", "-laws-cpp-sdk-core", "-laws-cpp-sdk-s3"],
    copts = ["-std=c++2a", "-fcoroutines"],
)


py_library(
    name = "pheonix",
    data = [":pheonix.so"],
)

py_test(
    name = "prefetch_simple",
    srcs = ["tests/prefetch_simple.py"],
    deps = [
        ":pheonix"
    ],
    python_version = "PY3",
)