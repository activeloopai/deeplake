import distutils.sysconfig as sysconfig
import json
import os
import sys
import platform
import requests

"""
Usage: python3 scripts/build_pg_ext.py debug            #Debug build
Usage: python3 scripts/build_pg_ext.py dev              #Develop build
Usage: python3 scripts/build_pg_ext.py prod             #Release build
"""

def download_static_lib(path_to_check):
    if os.path.exists(path_to_check):
        print("Static libraries already exist. Skipping download.")
        return

    # create directory for the deeplake static library if it does not exist
    deeplake_static_lib_dir = os.path.dirname(path_to_check)
    os.makedirs(deeplake_static_lib_dir, exist_ok=True)

    machine = platform.machine()

    # Get latest release from GitHub
    api_url = "https://api.github.com/repos/activeloopai/deeplake/releases/latest"
    print(f"Fetching latest release from {api_url} ...")

    response = requests.get(api_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch latest release info. Status code: {response.status_code}")

    release_data = response.json()
    tag_name = release_data.get("tag_name")
    if not tag_name:
        raise Exception("Failed to get tag_name from latest release")

    print(f"Latest release: {tag_name}")

    # Strip 'v' prefix from tag name if present (e.g., v4.4.2 -> 4.4.2)
    version = tag_name.lstrip('v')

    # Construct asset name based on platform
    archive_name = f"deeplake-static-{version}-linux-{machine}"
    zip_archive_name = f"{archive_name}.zip"
    tar_archive_name = f"{archive_name}.tar.gz"

    # Find the matching asset in the release
    asset_url = None
    for asset in release_data.get("assets", []):
        if asset.get("name") == zip_archive_name:
            asset_url = asset.get("browser_download_url")
            break

    if not asset_url:
        raise Exception(f"Could not find asset '{zip_archive_name}' in latest release")

    print(f"Downloading prebuilt static libraries from {asset_url} ...")

    response = requests.get(asset_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download static libraries from {asset_url}. Status code: {response.status_code}")

    with open(f"{zip_archive_name}", "wb") as f:
        f.write(response.content)

    err = os.system(f"unzip {zip_archive_name} -d /tmp && rm {zip_archive_name}")
    if err:
        raise Exception(f"Failed to extract static libraries. Command exited with code {err}.")

    err = os.system(f"tar -xzf /tmp/{tar_archive_name} -C /tmp && rm /tmp/{tar_archive_name}")
    if err:
        raise Exception(f"Failed to extract static libraries. Command exited with code {err}.")

    extracted_path = f"/tmp/{archive_name}"
    err = os.system(f"mv {extracted_path}/lib/* {deeplake_static_lib_dir} && rm -rf {extracted_path}")
    if err:
        raise Exception(f"Failed to move static libraries. Command exited with code {err}.")

def run(mode: str, incremental: bool):
    modes = ["debug", "dev", "prod"]
    if mode not in modes:
        raise Exception(f"Invalid mode - '{mode}'. Possible values - {', '.join(modes)}")

    os.chdir("cpp")

    # include path to specific verison from manylinux docker container in the path, so vcpkg_find_acquire_package can find it
    os.putenv("PATH", os.getenv("PATH")+f":/opt/_internal/cpython-{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}/bin/")

    preset = f"--preset=deeplake-pg-{mode}-windows" if platform.system() == "Windows" else f"--preset=deeplake-pg-{mode}"

    try:
        if not incremental:
            cmake_cmd = (f"cmake "
                         f"{preset} ")

            architectures = os.environ.get("CMAKE_OSX_ARCHITECTURES", "")
            if architectures:
                cmake_cmd += f"-D CMAKE_OSX_ARCHITECTURES={architectures} "

            err = os.system(cmake_cmd)
            if err:
                raise Exception(
                    f"Cmake command failed with exit code {err}. Full command: `{cmake_cmd}`"
                )

            download_static_lib(".ext/deeplake_static/lib/libdeeplake_static.a")

        make_cmd = f"cmake --build {preset}"
        err = os.system(make_cmd)
        if err:
            raise Exception(f"Command `{make_cmd}` failed with exit code {err}.")

    finally:
        os.chdir("..")
        write_mode(mode)

def read_mode():
    try:
        with open('.buildinfo') as f:
            try:
                data = json.load(f)
            except Exception as e:
                raise Exception("No previous mode found for incremental build. Please run full build.")

        if "mode" in data["deeplake-pg"]:
            return data["deeplake-pg"]["mode"]
    except Exception as e:
        raise Exception("No previous mode found for incremental build. Please run full build.")


def write_mode(mode: str):
    data = dict()
    try:
        with open('.buildinfo', 'r') as f:
            try:
                data = json.load(f)
            except Exception as e:
                pass
    except Exception as e:
        pass

    if not "deeplake-pg" in data.keys():
        data["deeplake-pg"] = dict()

    data["deeplake-pg"]["mode"] = mode

    with open('.buildinfo', 'w') as f:
        f.write(json.dumps(data))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run(mode=read_mode(), incremental=True)
    else:
        if len(sys.argv) == 2:
            run(mode=sys.argv[1], incremental=False)
        else:
            raise Exception("Invalid command line options")
