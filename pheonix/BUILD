load("@rules_cc//cc:defs.bzl", "cc_library")

cc_binary(
    name = "pheonix",
    srcs = ["eventuals.cpp"],
    deps = [
        "@com_github_3rdparty_eventuals//eventuals",
    ],
    # NOTE: at least on macOS we can't link statically because there
    # are certain system libraries that require dynamic loading.
    linkstatic = False,
)
