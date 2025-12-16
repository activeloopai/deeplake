import distutils.sysconfig as sysconfig
import json
import os
import sys
import platform
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
"""

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
            print(f"Found existing version {existing_version}, but latest is {version}. Downloading latest...")

    # Construct asset name based on platform
    archive_name = f"deeplake-api-{version}-linux-{machine}"
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

def run(mode: str, incremental: bool, deeplake_link_type: str = None, pg_versions: list = None):
    modes = ["debug", "dev", "prod"]
    if mode not in modes:
        raise Exception(f"Invalid mode - '{mode}'. Possible values - {', '.join(modes)}")

    os.chdir("cpp")

    # include path to specific verison from manylinux docker container in the path, so vcpkg_find_acquire_package can find it
    os.putenv("PATH", os.getenv("PATH")+f":/opt/_internal/cpython-{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}/bin/")

    preset = f"--preset=deeplake-pg-{mode}-windows" if platform.system() == "Windows" else f"--preset=deeplake-pg-{mode}"

    try:
        if not incremental:
            # Download DeepLake API library before CMake configuration
            # (CMake's find_package needs it to exist during configuration)
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
                raise Exception(f"Invalid option '{arg}'. Use --deeplake-shared, --deeplake-static, or --pg-versions")

        run(mode=mode, incremental=False, deeplake_link_type=deeplake_link_type, pg_versions=pg_versions)
