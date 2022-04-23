"""
Introduce protoc eventuals plugin.
"""

load("@com_github_reboot_dev_pyprotoc_plugin//:rules.bzl", "create_protoc_plugin_rule")

cc_eventuals_library = create_protoc_plugin_rule(
    "@com_github_3rdparty_eventuals//protoc-gen-eventuals:protoc-gen-eventuals",
    extensions = (".eventuals.h", ".eventuals.cc"),
)
