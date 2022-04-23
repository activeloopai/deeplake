"""Dependency specific initialization."""

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("@com_github_3rdparty_bazel_rules_asio//bazel:deps.bzl", asio_deps = "deps")
load("@com_github_3rdparty_bazel_rules_curl//bazel:deps.bzl", curl_deps = "deps")
load("@com_github_3rdparty_bazel_rules_jemalloc//bazel:deps.bzl", jemalloc_deps = "deps")
load("@com_github_3rdparty_bazel_rules_libuv//bazel:deps.bzl", libuv_deps = "deps")
load("@com_github_3rdparty_stout_borrowed_ptr//bazel:deps.bzl", stout_borrowed_ptr_deps = "deps")
load("@com_github_3rdparty_stout_notification//bazel:deps.bzl", stout_notification_deps = "deps")
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
load("@com_github_reboot_dev_pyprotoc_plugin//bazel:deps.bzl", pyprotoc_plugin_deps = "deps")

def deps(repo_mapping = {}):
    """Propagate all dependencies.

    Args:
        repo_mapping (str): {}.
    """
    asio_deps(
        repo_mapping = repo_mapping,
    )

    bazel_skylib_workspace()

    curl_deps(
        repo_mapping = repo_mapping,
    )

    jemalloc_deps(
        repo_mapping = repo_mapping,
    )

    libuv_deps(
        repo_mapping = repo_mapping,
    )

    maybe(
        http_archive,
        name = "platforms",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/0.0.5/platforms-0.0.5.tar.gz",
            "https://github.com/bazelbuild/platforms/releases/download/0.0.5/platforms-0.0.5.tar.gz",
        ],
        sha256 = "379113459b0feaf6bfbb584a91874c065078aa673222846ac765f86661c27407",
        repo_mapping = repo_mapping,
    )

    maybe(
        http_archive,
        name = "com_github_gflags_gflags",
        url = "https://github.com/gflags/gflags/archive/v2.2.2.tar.gz",
        sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
        strip_prefix = "gflags-2.2.2",
        repo_mapping = repo_mapping,
    )

    # NOTE: using glog version 0.5.0 since older versions failed
    # to compile on Windows, see:
    # https://github.com/google/glog/issues/472
    maybe(
        http_archive,
        name = "com_github_google_glog",
        url = "https://github.com/google/glog/archive/refs/tags/v0.5.0.tar.gz",
        sha256 = "eede71f28371bf39aa69b45de23b329d37214016e2055269b3b5e7cfd40b59f5",
        strip_prefix = "glog-0.5.0",
        repo_mapping = repo_mapping,
    )

    maybe(
        http_archive,
        name = "com_github_google_googletest",
        url = "https://github.com/google/googletest/archive/release-1.11.0.tar.gz",
        sha256 = "b4870bf121ff7795ba20d20bcdd8627b8e088f2d1dab299a031c1034eddc93d5",
        strip_prefix = "googletest-release-1.11.0",
        repo_mapping = repo_mapping,
    )

    stout_borrowed_ptr_deps(
        repo_mapping = repo_mapping,
    )

    stout_notification_deps(
        repo_mapping = repo_mapping,
    )

    pyprotoc_plugin_deps(
        repo_mapping = repo_mapping,
    )

    grpc_deps()
