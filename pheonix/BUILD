load("@rules_cc//cc:defs.bzl", "cc_library")
load("//3rdparty/eventuals:copts.bzl", "copts")

cc_binary(
    name = "pheonix",
    srcs = ["eventuals.cpp"],
    deps = [
        "@com_github_3rdparty_eventuals//:eventuals",
    ],
    copts = copts(),
)