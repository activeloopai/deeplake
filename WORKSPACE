workspace(name = "com_github_activeloopai_hub")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

########################################################################

# Pull in a recent version of abseil that correctly builds on macOS.

http_archive(
    name = "com_google_absl",
    url = "https://github.com/abseil/abseil-cpp/archive/refs/tags/20211102.0.tar.gz",
    strip_prefix = "abseil-cpp-20211102.0",
    sha256 = "dcf71b9cba8dc0ca9940c4b316a0c796be8fab42b070bb6b7cab62b48f0e66c4",
)

########################################################################

# NOTE: we pull in all stout-* repos as a local repositories pointing
# to the git submodules that we have so that we can do more efficient
# development between the repositories. We'll remove this for releases!

local_repository(
    name = "com_github_3rdparty_stout_atomic_backoff",
    path = "submodules/eventuals/submodules/stout/stout-atomic-backoff",
)

load("@com_github_3rdparty_stout_atomic_backoff//bazel:repos.bzl", stout_atomic_backoff_repos = "repos")

stout_atomic_backoff_repos(external = False)

local_repository(
    name = "com_github_3rdparty_stout_stateful_tally",
    path = "submodules/eventuals/submodules/stout/stout-stateful-tally",
)

load("@com_github_3rdparty_stout_stateful_tally//bazel:repos.bzl", stout_stateful_tally = "repos")

stout_stateful_tally(external = False)

local_repository(
    name = "com_github_3rdparty_stout_borrowed_ptr",
    path = "submodules/eventuals/submodules/stout/stout-borrowed-ptr",
)

load("@com_github_3rdparty_stout_borrowed_ptr//bazel:repos.bzl", stout_borrowed_ptr_repos = "repos")

stout_borrowed_ptr_repos(external = False)

local_repository(
    name = "com_github_3rdparty_stout_flags",
    path = "submodules/eventuals/submodules/stout/stout-flags",
)

load("@com_github_3rdparty_stout_flags//bazel:repos.bzl", stout_flags_repos = "repos")

stout_flags_repos(external = False)

local_repository(
    name = "com_github_3rdparty_stout_notification",
    path = "submodules/eventuals/submodules/stout/stout-notification",
)

load("@com_github_3rdparty_stout_notification//bazel:repos.bzl", stout_notification_repos = "repos")

stout_notification_repos(external = False)

local_repository(
    name = "com_github_3rdparty_stout",
    path = "submodules/eventuals/submodules/stout",
)

load("@com_github_3rdparty_stout//bazel:repos.bzl", stout_repos = "repos")

stout_repos(external = False)

########################################################################

# NOTE: we pull in 'pyprotoc-plugin' as a local repository pointing to the
# git submodule that we have so that we can do more efficient
# development between the two repositories. We'll remove this for
# releases!
local_repository(
    name = "com_github_reboot_dev_pyprotoc_plugin",
    path = "submodules/eventuals/submodules/pyprotoc-plugin",
)

load("@com_github_reboot_dev_pyprotoc_plugin//bazel:repos.bzl", pyprotoc_plugin_repos = "repos")

pyprotoc_plugin_repos(external = False)

########################################################################

# NOTE: we pull in 'eventuals' as a local repository pointing to the
# git submodule that we have so that we can do do more efficient
# development between the two repositories. We'll remove this for
# releases!
local_repository(
    name = "com_github_3rdparty_eventuals",
    path = "submodules/eventuals",
)

load("@com_github_3rdparty_eventuals//bazel:repos.bzl", eventuals_repos = "repos")

eventuals_repos(external = False)

########################################################################

# Now get all of eventuals deps (and recursively its deps' deps, etc).

load("@com_github_3rdparty_eventuals//bazel:deps.bzl", eventuals_deps = "deps")

eventuals_deps()

########################################################################

# And also grab grpc's extra deps.

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

grpc_extra_deps()

########################################################################

# grab pybind11

http_archive(
  name = "pybind11_bazel",
  strip_prefix = "pybind11_bazel-72cbbf1fbc830e487e3012862b7b720001b70672",
  urls = ["https://github.com/pybind/pybind11_bazel/archive/72cbbf1fbc830e487e3012862b7b720001b70672.zip"],
)
# We still require the pybind library.
http_archive(
  name = "pybind11",
  build_file = "@pybind11_bazel//:pybind11.BUILD",
  strip_prefix = "pybind11-2.9.2",
  urls = ["https://github.com/pybind/pybind11/archive/v2.9.2.tar.gz"],
)
load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")

########################################################################