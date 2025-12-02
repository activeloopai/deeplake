#!/usr/bin/env python3

import os
import sys
import json
import subprocess


def check_version_updated(*args):
    if args[0][0] != "release":
        print('Skipping version check, it is not a "release"')
        sys.exit(0)
    latest_version: bytes = subprocess.check_output(
        ["curl", "https://app.activeloop.ai/visualizer/version"]
    )
    latest_version = latest_version.decode()
    latest_version = latest_version.replace("'", "")
    latest_version = latest_version.replace('"', "")
    latest_semver = [int(i) for i in latest_version.split(".")]
    with open("package.json") as file:
        package_json = json.loads(file.read())
    new_version = package_json["version"]
    new_semver = [int(i) for i in new_version.split(".")]
    if new_version == latest_version:
        print("FAIL version is not updated")
        sys.exit(1)
    print(f"new version: {new_version}\nlatest version: {latest_version}")
    semver = {"MAJOR": 0, "MINOR": 1, "PATCH": 2}

    for key, value in semver.items():
        if new_semver[value] > latest_semver[value]:
            print(f"PASSED with {key} version: new version is higher than the latest.")
            sys.exit(0)
        elif new_semver[value] < latest_semver[value]:
            print(
                f"FAIL with {key} version: latest version is higher than new version."
            )
            sys.exit(1)


if __name__ == "__main__":
    Function = sys.argv[1]
    Args = sys.argv[2:]
    eval(Function + "(Args)")
