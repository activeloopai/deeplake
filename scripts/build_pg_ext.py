#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.28",
# ]
# ///

import json
import os
import sys
import platform

try:
    import requests
except ImportError:
    os.system("pip install requests --user --break-system-packages")
    import requests

"""
Usage: python3 scripts/build_pg_ext.py debug                               #Debug build
Usage: python3 scripts/build_pg_ext.py dev                                 #Develop build
Usage: python3 scripts/build_pg_ext.py prod                                #Release build
Usage: python3 scripts/build_pg_ext.py debug --deeplake-shared             #Debug build with shared deeplake_api linking
Usage: python3 scripts/build_pg_ext.py debug --deeplake-static             #Debug build with static deeplake_api linking (force)
Usage: python3 scripts/build_pg_ext.py dev --pg-versions 16,17,18          #Build for PostgreSQL 16, 17, and 18
Usage: python3 scripts/build_pg_ext.py dev --pg-versions 16                #Build for PostgreSQL 16 only
Usage: python3 scripts/build_pg_ext.py prod --pg-versions all              #Build for all supported PostgreSQL versions
Usage: python3 scripts/build_pg_ext.py dev --local-api /path/to/package    #Use local deeplake API package instead of downloading
"""




def get_pinned_version():
    """
    Read the pinned deeplake API version from DEEPLAKE_API_VERSION file.
    """
    # Look for version file in repo root (one level up from scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    version_file = os.path.join(repo_root, "DEEPLAKE_API_VERSION")

    if not os.path.exists(version_file):
        raise Exception(f"Version file not found: {version_file}")

    with open(version_file, 'r') as f:
        version = f.read().strip()

    if not version:
        raise Exception(f"Version file is empty: {version_file}")

    return version


def download_api_lib(api_root_dir, overwrite=True):
    """
    Download and extract the full deeplake API library including:
    - include/ headers
    - lib/ libraries
    - cmake configurations
    - pkg-config files
    """
    # create directory for the deeplake api library if it does not exist
    os.makedirs(api_root_dir, exist_ok=True)

    machine = platform.machine()

    # Get pinned version from DEEPLAKE_API_VERSION file
    version = get_pinned_version()
    print(f"Using pinned deeplake API version: {version}")

    # Check if library already exists
    lib_dir = os.path.join(api_root_dir, "lib")
    version_marker = os.path.join(api_root_dir, ".version")

    if os.path.exists(version_marker) and not overwrite:
        with open(version_marker, 'r') as f:
            existing_version = f.read().strip()
        if existing_version == version:
            print(f"Library version {version} already exists. Skipping download.")
            return
        else:
            print(f"Found existing version {existing_version}, but pinned version is {version}. Downloading...")

    # Construct asset name based on platform
    archive_name = f"deeplake-api-{version}-linux-{machine}"
    zip_archive_name = f"{archive_name}.zip"
    tar_archive_name = f"{archive_name}.tar.gz"

    # Construct download URL directly (GitHub releases follow a predictable URL pattern)
    asset_url = f"https://github.com/activeloopai/deeplake/releases/download/v{version}/{zip_archive_name}"

    print(f"Downloading prebuilt api libraries from {asset_url} ...")

    
    response = requests.get(asset_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download api libraries from {asset_url}. Status code: {response.status_code}")

    with open(f"{zip_archive_name}", "wb") as f:
        f.write(response.content)

    print("Extracting archives...")
    err = os.system(f"unzip -o {zip_archive_name} -d /tmp && rm {zip_archive_name}")
    if err:
        raise Exception(f"Failed to extract api libraries. Command exited with code {err}.")

    err = os.system(f"tar -xzf /tmp/{tar_archive_name} -C /tmp && rm /tmp/{tar_archive_name}")
    if err:
        raise Exception(f"Failed to extract api libraries. Command exited with code {err}.")

    extracted_path = f"/tmp/{archive_name}"

    # Remove existing library directory if overwrite is enabled
    if overwrite:
        print(f"Removing existing library at {api_root_dir}...")
        err = os.system(f"rm -rf {api_root_dir}/*")
        if err:
            raise Exception(f"Failed to remove existing library. Command exited with code {err}.")

    # Copy entire library structure (include/, lib/, cmake configs, pkg-config, etc.)
    print(f"Installing full library structure to {api_root_dir}...")
    err = os.system(f"cp -r {extracted_path}/* {api_root_dir}/ && rm -rf {extracted_path}")
    if err:
        raise Exception(f"Failed to copy api library structure. Command exited with code {err}.")

    # Write version marker
    with open(version_marker, 'w') as f:
        f.write(version)

    print(f"Successfully installed deeplake API library version {version}")


def install_local_api_lib(api_root_dir, local_path):
    """
    Install the deeplake API library from a local package directory (e.g. from an indra build).
    This copies include/, lib/, and cmake/ from the local package into .ext/deeplake_api/.
    """
    if not os.path.isdir(local_path):
        raise Exception(f"Local API path does not exist: {local_path}")

    # Verify the local package has the expected structure
    local_lib = os.path.join(local_path, "lib")
    local_include = os.path.join(local_path, "include")
    if not os.path.isdir(local_lib) or not os.path.isdir(local_include):
        raise Exception(f"Local API path missing lib/ or include/ directories: {local_path}")

    os.makedirs(api_root_dir, exist_ok=True)

    print(f"Installing local deeplake API from {local_path} ...")

    # Remove existing and copy from local
    err = os.system(f"rm -rf {api_root_dir}/*")
    if err:
        raise Exception(f"Failed to clean {api_root_dir}. Command exited with code {err}.")

    err = os.system(f"cp -r {local_path}/* {api_root_dir}/")
    if err:
        raise Exception(f"Failed to copy local API library. Command exited with code {err}.")

    print(f"Successfully installed local deeplake API from {local_path}")


def run(mode: str, incremental: bool, deeplake_link_type: str = None, pg_versions: list = None, local_api_path: str = None):
    modes = ["debug", "dev", "prod"]
    if mode not in modes:
        raise Exception(f"Invalid mode - '{mode}'. Possible values - {', '.join(modes)}")

    os.chdir("cpp")

    # include path to specific verison from manylinux docker container in the path, so vcpkg_find_acquire_package can find it
    os.putenv("PATH", os.getenv("PATH")+f":/opt/_internal/cpython-{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}/bin/")

    preset = f"--preset=deeplake-pg-{mode}-windows" if platform.system() == "Windows" else f"--preset=deeplake-pg-{mode}"

    try:
        if not incremental:
            # Install DeepLake API library before CMake configuration
            # (CMake's find_package needs it to exist during configuration)
            if local_api_path:
                install_local_api_lib(".ext/deeplake_api", local_api_path)
            else:
                download_api_lib(".ext/deeplake_api")

            cmake_cmd = (f"cmake "
                         f"{preset} ")

            architectures = os.environ.get("CMAKE_OSX_ARCHITECTURES", "")
            if architectures:
                cmake_cmd += f"-D CMAKE_OSX_ARCHITECTURES={architectures} "

            # Add PostgreSQL version options if specified
            if pg_versions is not None:
                supported_versions = {16, 17, 18}
                # Set all versions to OFF by default
                for ver in supported_versions:
                    if ver in pg_versions:
                        cmake_cmd += f"-D BUILD_PG_{ver}=ON "
                    else:
                        cmake_cmd += f"-D BUILD_PG_{ver}=OFF "

            # Add deeplake linking type option if specified
            if deeplake_link_type == "shared":
                cmake_cmd += "-D USE_DEEPLAKE_SHARED=ON "
            elif deeplake_link_type == "static":
                cmake_cmd += "-D USE_DEEPLAKE_SHARED=OFF "
            # If None, let CMake auto-detect based on file existence

            err = os.system(cmake_cmd)
            if err:
                raise Exception(
                    f"Cmake command failed with exit code {err}. Full command: `{cmake_cmd}`"
                )

        make_cmd = f"cmake --build {preset}"
        err = os.system(make_cmd)
        if err:
            raise Exception(f"Command `{make_cmd}` failed with exit code {err}.")

        # Install the built extension to PostgreSQL directories
        preset_name = f"deeplake-pg-{mode}-windows" if platform.system() == "Windows" else f"deeplake-pg-{mode}"
        install_dir = f"../builds/{preset_name}"
        install_cmd = f"cmake --install {install_dir}"
        print(f"Installing extension to PostgreSQL directories...")
        err = os.system(install_cmd)
        if err:
            raise Exception(f"Command `{install_cmd}` failed with exit code {err}.")

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
        mode = sys.argv[1]
        deeplake_link_type = None
        pg_versions = None
        local_api_path = None

        # Parse optional flags
        i = 2
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg == "--deeplake-shared":
                deeplake_link_type = "shared"
                i += 1
            elif arg == "--deeplake-static":
                deeplake_link_type = "static"
                i += 1
            elif arg == "--local-api":
                if i + 1 >= len(sys.argv):
                    raise Exception("--local-api requires a path to the local deeplake API package directory")
                local_api_path = os.path.abspath(sys.argv[i + 1])
                i += 2
            elif arg == "--pg-versions":
                if i + 1 >= len(sys.argv):
                    raise Exception("--pg-versions requires a value (e.g., '16,17,18' or 'all')")
                versions_str = sys.argv[i + 1]
                if versions_str == "all":
                    pg_versions = [16, 17, 18]
                else:
                    try:
                        pg_versions = [int(v.strip()) for v in versions_str.split(',')]
                        # Validate versions
                        supported = {16, 17, 18}
                        invalid = set(pg_versions) - supported
                        if invalid:
                            raise Exception(f"Invalid PostgreSQL versions: {invalid}. Supported: {supported}")
                    except ValueError:
                        raise Exception(f"Invalid --pg-versions format: '{versions_str}'. Use comma-separated numbers (e.g., '16,17,18') or 'all'")
                i += 2
            else:
                raise Exception(f"Invalid option '{arg}'. Use --deeplake-shared, --deeplake-static, --local-api, or --pg-versions")

        run(mode=mode, incremental=False, deeplake_link_type=deeplake_link_type, pg_versions=pg_versions, local_api_path=local_api_path)
