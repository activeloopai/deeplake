#!/bin/bash

set -eo pipefail

# Define global variables
DEB_PATH=$(mktemp -dt pg-deeplake-XXXXXX)
LOG_FILE="update-deb.log"

# Define text styles
YELLOW="\e[93m"
GREEN="\e[92m"
RED="\e[91m"
DEFAULT="\e[0m"

# Function to extract version using pkg-config
extract_version_from_pkgconfig() {
    local pkgconfig_path="cpp/.ext/deeplake_api/lib/pkgconfig"

    if [ ! -d "$pkgconfig_path" ]; then
        echo -e "${RED}Error: pkg-config directory not found at $pkgconfig_path${DEFAULT}" >&2
        echo -e "${RED}Make sure the DeepLake API library is downloaded first.${DEFAULT}" >&2
        exit 1
    fi

    # Set PKG_CONFIG_PATH and query version
    local version=$(PKG_CONFIG_PATH="$pkgconfig_path" pkg-config --modversion deeplake_api 2>/dev/null)

    if [ -z "$version" ]; then
        echo -e "${RED}Error: Could not extract version using pkg-config${DEFAULT}" >&2
        exit 1
    fi

    echo "$version"
}

# Define usage function
usage() {
    echo -e "Usage: bash $0 [version-release] <repository> <architecture> <gpg-keyid> <supported-versions>"
    echo -e "\nArguments:"
    echo -e "  version-release:    Optional. Version with release suffix (e.g., 4.4.4-1)."
    echo -e "                      If not provided, version is extracted from DeepLake API pkg-config."
    echo -e "  repository:         Repository directory path"
    echo -e "  architecture:       amd64 or arm64"
    echo -e "  gpg-keyid:          GPG key ID for signing"
    echo -e "  supported-versions: Comma-separated PostgreSQL versions (e.g., 16,17,18)"
    echo -e "\nExamples:"
    echo -e "  bash $0 4.4.4-1 /tmp/repo arm64 1F8B584DBEA11E9D 16,17,18"
    echo -e "  bash $0 /tmp/repo arm64 1F8B584DBEA11E9D 16,17,18  # Auto-detect version"
}

# Error handling function
if_failed() {
    if [ $? -ne 0 ]; then
        touch $LOG_FILE
        echo -e "${RED}$1${DEFAULT}" | tee -a $LOG_FILE
        exit 1
    fi
}

# Parse arguments (support both 4 and 5 argument formats)
if [ "$#" -eq 4 ]; then
    # Auto-extract version from pkg-config
    echo -e "${YELLOW}Extracting version from DeepLake API pkg-config...${DEFAULT}"
    AUTO_VERSION=$(extract_version_from_pkgconfig)
    VERSION="${AUTO_VERSION}-1"  # Default to -1 release suffix
    echo -e "${GREEN}Detected version: ${VERSION}${DEFAULT}\n"
    REPOSITORY=$1
    ARCH=$2
    GPG_KEY=$3
    SUPPORTED_VERSIONS_ARG=$4
elif [ "$#" -eq 5 ]; then
    # Use provided version
    VERSION=$1
    REPOSITORY=$2
    ARCH=$3
    GPG_KEY=$4
    SUPPORTED_VERSIONS_ARG=$5
else
    usage && exit 1
fi

CLEAN_VERSION=${VERSION/-*/}
readarray -t -d',' SUPPORTED_VERSIONS < <(printf "%s" "$SUPPORTED_VERSIONS_ARG")

# Determine artifact directory
if [ "${ARCH}" == "amd64" ]; then
    ARTIFACT_DIR="postgres-x86_64"
elif [ "${ARCH}" == "arm64" ]; then
    ARTIFACT_DIR="postgres-aarch64"
fi

# Create repo directory
mkdir -p "${REPOSITORY}/deb/pg-deeplake/pool/main/"

for version in "${SUPPORTED_VERSIONS[@]}"; do

    # Copy build artifacts
    echo -e "${YELLOW}Copying build artifacts...${DEFAULT}"
    mkdir -p "${DEB_PATH}/pg-deeplake-${version}/DEBIAN/"
    mkdir -p "${DEB_PATH}/pg-deeplake-${version}/usr/lib/postgresql/${version}/lib/"
    mkdir -p "${DEB_PATH}/pg-deeplake-${version}/usr/share/postgresql/${version}/extension/"
    install -m 644 "${ARTIFACT_DIR}"/pg_deeplake_"${version}".so "${DEB_PATH}/pg-deeplake-${version}/usr/lib/postgresql/${version}/lib/pg_deeplake"
    install -m 644 "${ARTIFACT_DIR}"/pg_deeplake*.sql            "${DEB_PATH}/pg-deeplake-${version}/usr/share/postgresql/${version}/extension/"
    install -m 644 "${ARTIFACT_DIR}"/pg_deeplake.control         "${DEB_PATH}/pg-deeplake-${version}/usr/share/postgresql/${version}/extension/"
    echo -e "${GREEN}Done.${DEFAULT}\n"

    # Define control file
    echo -e "${YELLOW}Generating the control file...${DEFAULT}"
    cat << EOF > "${DEB_PATH}/pg-deeplake-${version}/DEBIAN/control"
Package: pg-deeplake-${version}
Version: ${CLEAN_VERSION}
Section: database
Priority: optional
Architecture: ${ARCH}
Maintainer: Activeloop SRE <sre@activeloop.dev>
Description: PostgreSQL ${version} DeepLake Extension.
EOF
    if_failed "Error while generating the control file."
    echo -e "${GREEN}Done.${DEFAULT}\n"

    # Build the deb package
    echo -e "${YELLOW}Building the deb package for ${ARCH} PostgreSQL ${version}...${DEFAULT}"
    rm -f pg-deeplake-"${version}"_"${VERSION}"_"${ARCH}".deb
    dpkg --build "${DEB_PATH}/pg-deeplake-${version}" "pg-deeplake-${version}_${VERSION}_${ARCH}.deb"
    mv -f "pg-deeplake-${version}_${VERSION}_${ARCH}.deb" "${REPOSITORY}/deb/pg-deeplake/pool/main/"
    echo -e "${GREEN}Done.${DEFAULT}\n"
done
if_failed "Error building the packages."

# Create the Packages file for the repository
echo -e "${YELLOW}Generating and gzipping the Packages file...${DEFAULT}"
mkdir -p "${REPOSITORY}/deb/pg-deeplake/dists/stable/main/binary-${ARCH}"
pushd "${REPOSITORY}/deb/pg-deeplake/" > /dev/null &&
dpkg-scanpackages --arch "${ARCH}" pool/ > dists/stable/main/binary-"${ARCH}"/Packages && popd > /dev/null || exit 1
gzip -fk9 "${REPOSITORY}/deb/pg-deeplake/dists/stable/main/binary-${ARCH}/Packages"
if_failed "Error while generating the Packages file."
echo -e "${GREEN}Done.${DEFAULT}\n"

# Function for generating hashes
do_hash() {
    HASH_NAME="$1"
    HASH_CMD="$2"
    echo "${HASH_NAME}:"
    while IFS= read -r -d '' f; do
        f=${f#./}
        case "$f" in
            "Release"|"Release.gpg"|"InRelease") continue ;;
        esac
        hash_value=$(${HASH_CMD} "$f" | awk '{print $1}')
        file_size=$(wc -c < "$f")
        echo " $hash_value $file_size $f"
    done < <(find ./ -type f -print0)
}

# Function for generating and signing the Release file
do_release() {
    local release_file="${REPOSITORY}/deb/pg-deeplake/dists/stable/Release"
    local inrelease_file="${REPOSITORY}/deb/pg-deeplake/dists/stable/InRelease"
    tmp_release=$(mktemp)

    echo -e "${YELLOW}Generating the Release file...${DEFAULT}"
    cat << EOF > "${tmp_release}"
Origin: Activeloop AI
Label: pg-deeplake
Suite: stable
Codename: stable
Version: ${CLEAN_VERSION}
Architectures: amd64 arm64
Components: main
Description: Activeloop Software Repository.
Date: $(date -Ru)
EOF
    pushd "${REPOSITORY}/deb/pg-deeplake/dists/stable" > /dev/null &&
    {
        do_hash "MD5Sum" "md5sum"
        do_hash "SHA1" "sha1sum"
        do_hash "SHA256" "sha256sum"
    } >> "${tmp_release}" && popd > /dev/null || exit 1
    cp "${tmp_release}" "${release_file}"

    echo -e "${YELLOW}Signing the Release file and creating the InRelease file...${DEFAULT}"
    rm -f "${release_file}.gpg" "${inrelease_file}"
    gpg --default-key "${GPG_KEY}" -abs < "${release_file}" > "${release_file}.gpg"
    gpg --default-key "${GPG_KEY}" -abs --clearsign < "${release_file}" > "${inrelease_file}"
}

do_release
if_failed "Error while generating the Release file."
echo -e "${GREEN}Done.${DEFAULT}\n"

# Export GPG key
mkdir -p "${REPOSITORY}"/keys/
gpg --export --armor "${GPG_KEY}" > "${REPOSITORY}"/keys/activeloop.asc
if_failed "Error while exporting the GPG key."
echo -e "${GREEN}Done.${DEFAULT}\n"
